import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import Linear
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding

from .tokenlearner import TokenLearner, TokenFuser
from .rcsample import Mlp

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayerMLP(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)


def _align_osz_mask(osz_mask, size, device, dtype):
    """Move the (B, 3, H, W) OSZ mask onto the BEV feature grid.

    AXIS CONTRACT (fixed 2026-08-13 — was transposed 90°, see ISSUE.md P0-3):
      * OSZ masks are exported with ``axis-0 = ego-x`` (forward, 0.15 m/cell)
        and ``axis-1 = ego-y`` (left, 0.3 m/cell) — OSZ/config.py and
        OSZ/modules/bev_height_builder.py (``xi`` indexes axis-0, ``yi``
        axis-1); the online ``build_osz_mask_online`` (torch_pipeline.py)
        follows the same convention.
      * The ResWorld BEV feature map is the OPPOSITE: ``axis-0 = ego-y`` and
        ``axis-1 = ego-x`` — rcsample.py::create_grid_infos builds
        ``bev_coor`` so the x component varies along axis-1 (W) and the y
        component along axis-0 (H); col_attn samples it with
        ``spatial_shapes=(bev_w, bev_h)`` (x along W).
      * The mask is therefore TRANSPOSED here (before the resize): OSZ cell
        (i=xi, j=yi) must land on feature cell (row=yi, col=xi) — i.e. M.T.
        Keep OSZ as ``axis-0=x``; do NOT "fix" OSZ to ``axis-0=y``, or this
        transpose double-flips the mask back into the bug.
    """
    if osz_mask.dim() != 4:
        raise ValueError(
            f"osz_mask must be (B, 3, H, W), got {tuple(osz_mask.shape)}")
    m = osz_mask.to(device=device, dtype=dtype)
    m = m.transpose(-1, -2)              # OSZ axis-0=x -> feature axis-0=y
    if m.shape[-2:] != size:
        m = F.interpolate(m, size=size, mode="nearest")
    return m


class OcclusionAwareFusion(nn.Module):
    """Stage-2 occlusion injection — three modes (V2, design doc Stage 2).

    ``off``           : no injection at all (planner BEV stays baseline-clean).
    ``raw_additive``  : legacy residual offset ``B~ = B + E_occ * M`` (V1 form,
                        kept as an ablation arm; zero-init offset, identity at
                        init).
    ``risk_gated``    : risk-gated back-flow
                        ``B~ = B + g (.) Proj([R_exp.detach(), R_worst.detach()])``
                        — the injected CONTENT is the safety-supervised
                        RiskHead reading (detached, ISSUE.md I2: the planning
                        loss cannot reshape it) and the BANDWIDTH is the
                        per-channel gate ``g`` trained by the planning loss
                        (ISSUE.md I3).

    Implementation contract (ISSUE.md 2.1, both written into the init):
      * ``gate_raw`` is ZERO-init and ``g = tanh(gate_raw)``, so ``|g| <= 1``
        holds by construction and step-0 injection is exactly 0 (I1:
        pointwise baseline-equal at init).
      * ``risk_proj`` keeps the default (Kaiming) gain — only ONE of the two
        may be zero-init, otherwise dL/dg = Proj(.) = 0 and dL/dProj ~ g = 0
        deadlock the gate forever (zero-multiply deadlock).
      * During the warmup period (``iters < gate_warmup_iters``) the gate is
        detached, so the injection value stays EXACTLY 0 while ``risk_proj``
        and ``gate_raw`` both stay in the autograd graph (proj through the
        zero-multiplied add, gate through the always-computed ``loss_gate``)
        — no DDP unused-parameter failure, no gate drift before the risk
        head has been shaped by its own supervision.
    """

    def __init__(self, in_channels: int, mask_channels: int = 3,
                 mode: str = 'off', gate_warmup_iters: int = 2000):
        super(OcclusionAwareFusion, self).__init__()
        assert mode in ('off', 'raw_additive', 'risk_gated'), \
            f"unknown osz_inject_mode: {mode!r}"
        self.mode = mode
        self.gate_warmup_iters = gate_warmup_iters
        if mode == 'raw_additive':
            # V1 branch: learned occlusion offset over the 3 mask channels,
            # gated by osz_eye; zero-init -> identity at init (ISSUE P0-4).
            self.osz_embed = nn.Sequential(
                nn.Conv2d(mask_channels, in_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            nn.init.zeros_(self.osz_embed[0].weight)
            nn.init.zeros_(self.osz_embed[0].bias)
        if mode == 'risk_gated':
            # 1x1 conv 2 -> C_bev with the DEFAULT (Kaiming) gain — the
            # gradient-hunger guard for the zero-init gate (ISSUE 2.1).
            self.risk_proj = nn.Conv2d(2, in_channels, kernel_size=1)
            self.gate_raw = nn.Parameter(torch.zeros(in_channels))

    def gate(self, iters=None):
        """Effective per-channel gate ``g = tanh(gate_raw)`` in [-1, 1].

        Detached (frozen at 0) during the warmup period so the injection
        stays exactly zero until the RiskHead output has been shaped by its
        own safety supervision. ``iters=None`` = outside training (eval):
        the gate is live.
        """
        g = torch.tanh(self.gate_raw)
        if iters is not None and iters < self.gate_warmup_iters:
            g = g.detach()
        return g

    def forward(self, bev, src, iters=None):
        """Inject into ``bev`` (B, C, H, W).

        ``src`` is the mask (B, 3, H, W) for ``raw_additive`` and the risk
        field (B, 2, H, W) for ``risk_gated``.
        """
        if self.mode == 'off':
            return bev
        if self.mode == 'raw_additive':
            m = src[:, :1]             # strict occlusion = osz_eye channel
            e_occ = self.osz_embed(src)   # (bs, C, h, w) learned offset
            return bev + e_occ * m
        # risk_gated: content detached (I2), bandwidth learned (I3).
        proj = self.risk_proj(src.detach())            # (B, C, H, W)
        g = self.gate(iters=iters)                     # (C,)
        return bev + proj * g.view(1, -1, 1, 1)


class RiskHead(nn.Module):
    """Stage-4 learnable risk head (V2, design doc Stage 4).

    All inputs arrive DETACHED (ISSUE.md I2 — the risk reading never
    back-shapes the perception / world-model / MHST trunks; ISSUE.md I4 —
    the auxiliary supervision L_col/L_dyn stops at this head):

      B_t        : (B, C, H, W)   current BEV feature (detached pred_bev)
      M          : (B, 1, H, W)   strict occlusion mask (osz_eye)
      drivable   : (B, 1, H, W)   HD-map drivable area (optional)
      phi        : (B, 1, H, W)   content-change intensity
                                   sigma/(1+sigma) * mean_k ||dB^k||_2
      s_occ      : (B, 1, H, W)   occlusion-occupancy head logits
      dist_inv   : (B, 1, H, W)   1/(1+d) proximity to nearest dynamic object
      cmd_embed  : (B, cmd_dim)   navi embedding shared with the planner
                                   (independent small projection)

    Output contract:
      mu         = sigmoid(logit)          risk mean (BCE-compatible, in
                                           (0,1) — the bounded, everywhere-
                                           differentiable counterpart of the
                                           design doc's softplus)
      sigma_epis = std over T MC-dropout forwards of the sigmoid output
      R_exp   = clamp(mu,                    0, risk_max)
      R_worst = clamp(mu + beta*sigma_epis,  0, risk_max)   # UCB upper bound

    MC-dropout (design doc 4.3): the last two conv layers carry dropout and
    ``mc_t`` stochastic forwards are averaged — mu = mean, sigma_epis = std.
    During eval the head runs a single forward (sigma_epis = 0, R_worst ==
    R_exp) unless ``mc_eval`` is set. The outputs pass a fixed Gaussian
    smoothing kernel (5x5, sigma 1.0) so grid_sample gradients stay stable
    (ISSUE.md I5: bounded UCB).
    """

    def __init__(self, bev_channels=256, cmd_dim=256, cmd_proj=8,
                 hidden=64, dropout=0.1, mc_t=4, mc_eval=False,
                 beta=1.0, risk_max=1.0, smooth_ks=5, smooth_sigma=1.0):
        super(RiskHead, self).__init__()
        extra = 5  # M + drivable + phi + s_occ + dist_inv
        self.conv1 = nn.Conv2d(bev_channels + extra + cmd_proj,
                               hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1)
        self.out = nn.Conv2d(hidden // 2, 1, kernel_size=1)
        # MC-dropout on the LAST TWO layers (design doc 4.3). Explicit
        # F.dropout2d with a `training` flag — a nn.Dropout2d module would
        # be inert under model.eval(), killing the mc_eval mode.
        self.drop_p = dropout
        self.cmd_proj = nn.Linear(cmd_dim, cmd_proj)
        self.beta = beta
        self.risk_max = risk_max
        self.mc_t = max(1, int(mc_t))
        self.mc_eval = mc_eval
        # Fixed Gaussian smoothing kernel (5x5, sigma) — not trained.
        ks = smooth_ks
        x = torch.arange(ks, dtype=torch.float32) - ks // 2
        kern = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2)
                         / (2.0 * smooth_sigma ** 2))
        self.register_buffer('smooth_kernel',
                             (kern / kern.sum()).view(1, 1, ks, ks))

    def _forward_once(self, x, cmd_feat, training):
        h = torch.cat([x, cmd_feat.expand(-1, -1, x.shape[2], x.shape[3])],
                      dim=1)
        h = F.relu(self.conv1(h), inplace=True)
        h = F.dropout2d(F.relu(self.conv2(h), inplace=True),
                        p=self.drop_p, training=training)
        h = F.dropout2d(F.relu(self.conv3(h), inplace=True),
                        p=self.drop_p, training=training)
        return self.out(h)                       # (B, 1, H, W) logit

    def forward(self, bev, mask, phi, s_occ, dist_inv,
                drivable=None, cmd_embed=None):
        """``bev``/``mask``/... must already be detached by the caller.

        Returns ``(field, stats)`` with ``field`` = (B, 2, H, W) stacked
        [R_exp, R_worst] and ``stats = {'mu': mu, 'sigma_epis': sigma_epis}``
        (smoothed).
        """
        xs = [bev, mask, phi, s_occ, dist_inv]
        if drivable is not None:
            xs.append(drivable)
        x = torch.cat(xs, dim=1)                 # (B, C+5(+1), H, W)
        cmd_feat = self.cmd_proj(cmd_embed).unsqueeze(-1).unsqueeze(-1) \
            if cmd_embed is not None else torch.zeros(
                x.shape[0], self.cmd_proj.out_features, 1, 1,
                device=x.device, dtype=x.dtype)
        # Fix cmd_feat dtype (x is fp32 under force_fp32).
        cmd_feat = cmd_feat.to(dtype=x.dtype)

        do_mc = self.training or self.mc_eval
        if do_mc and self.mc_t > 1:
            ps = [torch.sigmoid(self._forward_once(x, cmd_feat, training=True))
                  for _ in range(self.mc_t)]
            p = torch.stack(ps, dim=0)           # (T, B, 1, H, W)
            mu = p.mean(dim=0)                   # (B, 1, H, W)
            sigma_epis = p.std(dim=0)
        else:
            mu = torch.sigmoid(
                self._forward_once(x, cmd_feat, training=False))
            sigma_epis = torch.zeros_like(mu)

        mu = self._smooth(mu)
        sigma_epis = self._smooth(sigma_epis)
        r_exp = mu.clamp(min=0.0, max=self.risk_max)
        r_worst = (mu + self.beta * sigma_epis).clamp(
            min=0.0, max=self.risk_max)
        field = torch.stack([r_exp, r_worst], dim=1)   # (B, 2, H, W)
        return field, {'mu': mu, 'sigma_epis': sigma_epis}

    def _smooth(self, t):
        return F.conv2d(t, self.smooth_kernel, padding=self.smooth_kernel.shape[-1] // 2)


class OcclusionMHSTHead(nn.Module):
    """Stage-3 multi-hypothesis stochastic transition head (OARWM).

    Grafted AFTER the ResWorld deterministic residual path (right after
    ``pred_bev = tokenfuser(...) + bev_navi_embed``): inside occluded cells
    (mask channel 0 = osz_eye) the single deterministic guess is replaced
    by a K-hypothesis mixture; visible cells stay untouched::

        B_hat(x,y) = pred_bev(x,y)                              visible
                   = sum_k pi_k(x,y) * (pred_bev + dB^k)(x,y)   occluded

    Components (design doc Stage 3):
      * prior network g_prior: [pred_bev | mask] -> 1x1 conv -> multi-scale
        dilated neighbourhood (3x3 convs, dilation 1/2/4, receptive fields
        3/5/9 cells, aggregating N(x,y)) -> concat -> 1x1 conv -> K logits
        -> softmax = pi (B, K, H, W).
      * K-hypothesis residuals (MoE-style): one shared backbone
        (1x1 + 3x3 conv) then K independent expert 3x3 convs -> dB^k
        (B, K, C, H, W). K=1 is the single-hypothesis ablation.
      * uncertainty: sigma = softplus(s) + sigma_min (B, 1, H, W), kept
        for the calibration losses; it does NOT alter the fused output
        here (the mixture already is the expectation).

    An all-zero mask makes the head the identity (fused == pred_bev), so
    ``use_oarwm=False`` (or an empty mask) stays strictly baseline-exact.
    """

    def __init__(self, in_channels, k=3, sigma_min=0.1, mask_channels=3,
                 hidden=128, delta_clamp=10.0, sigma_max=10.0):
        super(OcclusionMHSTHead, self).__init__()
        self.k = k
        self.sigma_min = sigma_min
        # ISSUE P0-6 (2026-08): hard bounds on the hypothesis residuals and
        # the uncertainty. A hypothesis whose posterior weight collapses to
        # ~0 loses its exposure-supervision gradient, so its dB drifts
        # upward freely and R_worst = max_k inherits the unbounded magnitude
        # (2026-08-13 training explosion). Normal ||dB|| per element is
        # ~1-3 (||dB||_2 ~ 16-40); clamp at +-10 leaves ~4x headroom.
        self.delta_clamp = delta_clamp
        # Normal sigma is ~0.1-1 (calibrated to the exposure error); the
        # cap keeps var bounded so the mixture-likelihood constraint on dB
        # (gradient ~ 1/var) never fades.
        self.sigma_max = sigma_max

        # g_prior: mask + BEV context -> pi over K hypotheses.
        # Multi-scale dilated neighbourhood (dilation 1/2/4 -> receptive
        # fields 3/5/9 cells) so the prior sees the occluded-boundary
        # context N(x,y) beyond the 3-cell limit of a single 3x3 conv
        # (design doc Stage 3 limitation -> implemented fix).
        self.prior_conv1 = nn.Conv2d(
            in_channels + mask_channels, hidden, kernel_size=1)
        self.prior_d1 = nn.Conv2d(hidden, hidden, kernel_size=3,
                                  padding=1, dilation=1)
        self.prior_d2 = nn.Conv2d(hidden, hidden, kernel_size=3,
                                  padding=2, dilation=2)
        self.prior_d4 = nn.Conv2d(hidden, hidden, kernel_size=3,
                                  padding=4, dilation=4)
        self.prior_out = nn.Conv2d(hidden * 3, k, kernel_size=1)
        # Shared residual backbone, then one expert head per hypothesis.
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.experts = nn.ModuleList([
            nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1)
            for _ in range(k)
        ])
        # Zero-init the expert output layers (ISSUE.md P1-2, fixed 2026-08):
        # at init dB^k == 0, so the occluded patch == pred_bev and the fused
        # output is exactly the baseline at step 0 — no random perturbation
        # pollutes col_attn during early training; the hypotheses grow
        # gradually under the Stage-6 losses.
        for expert in self.experts:
            nn.init.zeros_(expert.weight)
            nn.init.zeros_(expert.bias)
        self.sigma_net = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, pred_bev, mask):
        m = mask[:, :1]                       # strict occlusion = osz_eye
        h0 = F.relu(
            self.prior_conv1(torch.cat([pred_bev, mask], dim=1)),
            inplace=True,
        )
        branches = [self.prior_d1(h0), self.prior_d2(h0), self.prior_d4(h0)]
        pi = self.prior_out(torch.cat(branches, dim=1))
        pi = pi.softmax(dim=1)                # (B, K, H, W)

        h = self.backbone(pred_bev)           # (B, hidden, H, W)
        delta = torch.stack(
            [expert(h) for expert in self.experts], dim=1)  # (B, K, C, H, W)
        if self.delta_clamp is not None and self.delta_clamp > 0:
            delta = delta.clamp(min=-self.delta_clamp, max=self.delta_clamp)

        occ = pred_bev.unsqueeze(1) + delta   # (B, K, C, H, W)
        patch = (occ * pi.unsqueeze(2)).sum(dim=1)          # (B, C, H, W)

        sigma = F.softplus(self.sigma_net(h)) + self.sigma_min
        if self.sigma_max is not None and self.sigma_max > 0:
            sigma = sigma.clamp(max=self.sigma_max)

        fused = pred_bev * (1 - m) + patch * m
        aux = {'pi': pi, 'sigma': sigma, 'delta': delta}
        return fused, aux


@HEADS.register_module()
class ResWorldHead(BaseModule):
    def __init__(self,
                #  *args,
                 grid_config,
                 num_frames=3,
                 embed_dims=256,
                 in_channels=256,
                 num_reg_fcs=2,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 num_scenes=16,
                 latent_decoder=None,
                 res_latent_decoder=None,
                 way_decoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 # Injection gate: True iff any mask source is active
                 # (config: use_osz = use_osz_midas or use_osz_rcsample).
                 use_osz=False,
                 # Stage-2 injection mode (V2, three-state): 'off' = no
                 # injection at all (baseline-equal planner BEV);
                 # 'raw_additive' = legacy V1 residual offset (ablation arm);
                 # 'risk_gated' = risk-gated back-flow B~ = B +
                 # g ⊙ Proj([R_exp.detach(), R_worst.detach()]).
                 osz_inject_mode='off',
                 # Gate warmup in ITERS: during warmup the gate is detached
                 # (injection value exactly 0) while Proj keeps its default
                 # Kaiming gain — zero-multiply deadlock / gradient hunger
                 # guards (ISSUE.md 2.1). Set from the runner iter count by
                 # CustomSetEpochInfoHook.before_train_iter.
                 gate_warmup_iters=2000,
                 # L1 bandwidth budget on g (design doc Stage 2, λ ~ 1e-5).
                 loss_gate_weight=1e-5,
                 # Stage-3 OARWM switch: multi-hypothesis transition head
                 # (see OARWM_ResWorld.md Stage 3).
                 use_oarwm=False,
                 mhst_k=3,
                 mhst_sigma_min=0.1,
                 # ISSUE P0-6 (2026-08): hard bounds against the
                 # zombie-hypothesis explosion (see OcclusionMHSTHead).
                 mhst_delta_clamp=10.0,
                 mhst_sigma_max=10.0,
                 # Stage-4 learnable risk head (V2, MC-dropout UCB).
                 use_risk_field=False,
                 risk_hidden=64,
                 risk_dropout=0.1,
                 risk_mc_t=4,
                 risk_mc_eval=False,
                 # UCB coefficient — PROBABILITY scale (V2): sigma_epis is
                 # the std of the sigmoid outputs (<= 0.5, typically
                 # 0.05-0.2), so beta=1 is a 1-sigma UCB bound and beta>2
                 # saturates against risk_max. Not comparable with the V1
                 # clamp-100 field scale.
                 risk_beta=1.0,
                 risk_max=1.0,
                 risk_smooth_ks=5,
                 risk_smooth_sigma=1.0,
                 risk_cmd_proj=8,
                 # Stage-6 loss weights (0 disables the term).
                 loss_div_weight=0.1,
                 loss_occ_halluc_weight=1.0,
                 loss_uncertainty_weight=1.0,
                 loss_occ_gt_weight=1.0,
                 loss_occ_gt_pos_weight=5.0,
                 # risk-field grounding (design doc §6.2c): anchors the
                 # raw risk intensity to the exposure error e2 (zero extra
                 # data). 0 disables the term.
                 loss_risk_ground_weight=0.1,
                 # Stage-4 safety supervision (V2): L_col / L_dyn train the
                 # RiskHead only (ISSUE I4 — they stop at the risk head).
                 loss_col_weight=1.0,
                 loss_col_pos_weight=10.0,
                 loss_dyn_weight=1.0,
                 loss_dyn_pos_weight=5.0,
                 # S_dyn forward margin along the object heading (m) and the
                 # dynamic-object velocity threshold (m/s) for
                 # _rasterise_dynamic.
                 dyn_forward_margin=2.0,
                 dyn_vel_thresh=0.5,
                 # Stage-5 risk-weighted planning (V2).
                 use_risk_plan=False,
                 # 'absolute_hinge' (main): L_plan_guard = mean_t max(0,
                 # R_worst(tau_t) - R_safe) with R_safe calibrated from the
                 # collision-positive mu quantile (EMA). 'gt_relative' (V1,
                 # fallback arm): along-path risk upper-bounded by the GT
                 # trajectory, max(0, R(pred) - R(gt) - margin).
                 risk_plan_mode='absolute_hinge',
                 loss_plan_guard_weight=0.1,
                 loss_plan_risk_weight=0.1,
                 loss_plan_cvar_weight=0.0,
                 cvar_beta=0.25,
                 # R_safe calibration: EMA_0.999[ quantile_q(mu | collision
                 # cells) ]. Buffer init 1.0 (guard dormant until calibrated).
                 risk_safe_quantile=0.1,
                 risk_safe_ema=0.999,
                 # Risk terms are disabled for the first N epochs (risk head
                 # randomly initialised — ISSUE.md P0-1/P1-2). The field is
                 # still sampled so the [DIAG] line keeps reporting r_cmd
                 # during warmup.
                 risk_plan_warmup_epochs=2,
                 # Linear ramp of the risk weights after warmup (epochs):
                 # the hard 0->1 switch made loss_plan_reg jump 0.53->1.24
                 # at the activation epoch (2026-08-13/14 log) — the ramp
                 # spreads the planner-vs-risk trade-off over N epochs so
                 # reg settles smoothly above init instead of spiking.
                 risk_plan_ramp_epochs=2,
                 # GT-risk upper-bound margin (gt_relative mode only):
                 # max(0, R(pred) - R(gt) - margin). 0 = pure upper bound.
                 risk_plan_margin=1.0,
                 **kwargs):
        super(ResWorldHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.latent_decoder = latent_decoder
        self.res_latent_decoder = res_latent_decoder
        self.way_decoder = way_decoder
        self.positional_encoding = positional_encoding
        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.num_frames = num_frames
        self.grid_min = torch.tensor([grid_config['x'][0], grid_config['y'][0]])
        self.grid_max = torch.tensor([grid_config['x'][1], grid_config['y'][1]])
        self.grid_size = torch.tensor([grid_config['x'][2], grid_config['y'][2]])

        # Stage-2 OSZ switch. When False the occlusion fusion module is NOT
        # created at all: strictly baseline (zero extra params, and no
        # DDP 'unused parameter' failure under find_unused_parameters=False).
        self.use_osz = use_osz
        self.osz_inject_mode = osz_inject_mode
        self.gate_warmup_iters = gate_warmup_iters
        self.loss_gate_weight = loss_gate_weight
        # Runner iter counter, injected by
        # CustomSetEpochInfoHook.before_train_iter (gate warmup, diag).
        # 10**9 = effectively "live gate" outside training (eval/test).
        self.iter = 10 ** 9
        # Stage-3 OARWM switch: same rationale — the MHST head is only
        # created when active, so the strict baseline has zero extra params
        # and no DDP unused-parameter failure.
        self.use_oarwm = use_oarwm
        self.mhst_k = mhst_k
        self.mhst_sigma_min = mhst_sigma_min
        self.mhst_delta_clamp = mhst_delta_clamp
        self.mhst_sigma_max = mhst_sigma_max
        self.use_risk_field = use_risk_field
        self.risk_hidden = risk_hidden
        self.risk_dropout = risk_dropout
        self.risk_mc_t = risk_mc_t
        self.risk_mc_eval = risk_mc_eval
        self.risk_beta = risk_beta
        self.risk_max = risk_max
        self.risk_smooth_ks = risk_smooth_ks
        self.risk_smooth_sigma = risk_smooth_sigma
        self.risk_cmd_proj = risk_cmd_proj
        # Stage-6 loss weights (per-term switch via weight=0).
        self.loss_div_weight = loss_div_weight
        self.loss_occ_halluc_weight = loss_occ_halluc_weight
        self.loss_uncertainty_weight = loss_uncertainty_weight
        self.loss_occ_gt_weight = loss_occ_gt_weight
        self.loss_occ_gt_pos_weight = loss_occ_gt_pos_weight
        self.loss_risk_ground_weight = loss_risk_ground_weight
        self.loss_col_weight = loss_col_weight
        self.loss_col_pos_weight = loss_col_pos_weight
        self.loss_dyn_weight = loss_dyn_weight
        self.loss_dyn_pos_weight = loss_dyn_pos_weight
        self.dyn_forward_margin = dyn_forward_margin
        self.dyn_vel_thresh = dyn_vel_thresh
        # Proximity dilation rings for _rasterise_dynamic's 1/(1+d).
        self._dist_levels = 10
        # Stage-5 risk-weighted planning.
        self.use_risk_plan = use_risk_plan
        self.risk_plan_mode = risk_plan_mode
        assert self.risk_plan_mode in ('absolute_hinge', 'gt_relative'), \
            f"unknown risk_plan_mode: {self.risk_plan_mode!r}"
        self.loss_plan_guard_weight = loss_plan_guard_weight
        self.loss_plan_risk_weight = loss_plan_risk_weight
        self.loss_plan_cvar_weight = loss_plan_cvar_weight
        self.cvar_beta = cvar_beta
        self.risk_safe_quantile = risk_safe_quantile
        self.risk_safe_ema = risk_safe_ema
        self.risk_plan_warmup_epochs = risk_plan_warmup_epochs
        self.risk_plan_ramp_epochs = risk_plan_ramp_epochs
        self.risk_plan_margin = risk_plan_margin
        # R_safe calibration state (design doc Stage 5): EMA over the
        # collision-positive mu quantile. Buffer = synced by DDP broadcast
        # (the EMA update itself is all_reduce'ed in loss()). Init 1.0 =
        # the guard is dormant until calibrated (R_worst <= risk_max = 1.0).
        self.register_buffer(
            'risk_safe', torch.tensor(1.0, dtype=torch.float32))
        self._init_layers()
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_reg_init = build_loss(loss_plan_reg)
                
    def _init_layers(self):
        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        init_ego_fut_decoder = []
        for _ in range(self.num_reg_fcs):
            init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            init_ego_fut_decoder.append(nn.ReLU())
        init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.init_ego_fut_decoder = nn.Sequential(*init_ego_fut_decoder)

        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayerMLP(self.embed_dims)
        self.canbus_mlp = Mlp(18, self.embed_dims, self.embed_dims)
        self.canbus_se = SELayerMLP(self.embed_dims)
        self.bev_fusion_conv = ConvModule(self.in_channels * self.num_frames, self.in_channels, 
                                          kernel_size=3, padding=1)
        if self.use_osz:
            # Three-state Stage-2 injection (V2). 'off' still creates the
            # module (zero params) so `hasattr(self, 'osz_fusion')` holds.
            self.osz_fusion = OcclusionAwareFusion(
                self.in_channels, mode=self.osz_inject_mode,
                gate_warmup_iters=self.gate_warmup_iters)
        if self.use_oarwm:
            self.mhst = OcclusionMHSTHead(
                self.in_channels, k=self.mhst_k,
                sigma_min=self.mhst_sigma_min,
                delta_clamp=self.mhst_delta_clamp,
                sigma_max=self.mhst_sigma_max)
            if self.loss_occ_gt_weight > 0 or self.use_risk_field:
                # Occlusion occupancy head: supervised by L_occ_gt (V2: the
                # rasterised DYNAMIC occupancy S_gt) and consumed by the
                # RiskHead as its s_occ input.
                self.occ_head = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        if self.use_risk_field:
            # Stage-4 learnable risk head (V2): MC-dropout UCB.
            self.risk_head = RiskHead(
                bev_channels=self.in_channels,
                cmd_dim=self.embed_dims, cmd_proj=self.risk_cmd_proj,
                hidden=self.risk_hidden, dropout=self.risk_dropout,
                mc_t=self.risk_mc_t, mc_eval=self.risk_mc_eval,
                beta=self.risk_beta, risk_max=self.risk_max,
                smooth_ks=self.risk_smooth_ks,
                smooth_sigma=self.risk_smooth_sigma)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.res_tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.tokenfuser = TokenFuser(self.num_scenes, 256)

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)
        self.res_latent_decoder = build_transformer_layer_sequence(self.res_latent_decoder)
        self.col_attn = MultiScaleDeformableAttention(self.embed_dims,
                                            num_points=8, num_levels=1)
        self.action_mln = MLN(6*2)
        self.positional_encoding = build_positional_encoding(
            self.positional_encoding)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.res_latent_decoder is not None:
            for p in self.res_latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 

    @force_fp32(apply_to=('bev_feats'))
    def forward(self,
                bev_inputs,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cmd=None,
                osz_mask=None,
                drivable_mask=None,
                gt_bboxes=None,
                gt_bev_next=None
            ):
        
        bev_feats, can_bus_infos = bev_inputs
        bt, c, h, w = bev_feats.shape
        bs = bt // self.num_frames
        dtype = bev_feats[0].dtype
        device = bev_feats[0].device
        can_bus_infos = self.canbus_mlp(can_bus_infos.permute(1, 0, 2)).view(bt, 1, self.in_channels)
        bev_embed = self.canbus_se(bev_feats.view(bt, c, h*w).permute(0, 2, 1), can_bus_infos)
        bev_embed_single = bev_embed.clone()
        bev_embed = bev_embed.permute(0, 2, 1).view(self.num_frames, bs, c, h, w).permute(1, 0, 2, 3, 4)
        bev_embed = self.bev_fusion_conv(bev_embed.reshape(bs, self.num_frames * c, h, w))
        if self.use_osz and self.osz_inject_mode == 'raw_additive' \
                and osz_mask is not None:
            # V1 ablation arm: learned residual occlusion offset on the
            # shared BEV (mask channels (osz_eye, osz_ground, semi)).
            # All-zeros mask = identity.
            m = _align_osz_mask(osz_mask, (h, w), device, dtype)
            bev_embed = self.osz_fusion(bev_embed, m)
        bev_feats = bev_feats.view(self.num_frames, bs, c, h, w)


        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_feats.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        bev_embed = bev_embed.reshape(bs, c, h * w).permute(0, 2, 1)
        # res_embed = res_embed.reshape(bs, c, h * w).permute(0, 2, 1)

        navi_embed = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            navi_embed.append(self.navi_embedding.weight[cmd_idx][None, None])
        navi_embed = torch.cat(navi_embed, dim=0)

        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)
        _, res_selected = self.res_tokenlearner(bev_query)
        bev_embed_single = torch.cat((bev_embed_single, pos_embd.repeat(self.num_frames,1,1)), -1)
        bev_embed_single = torch.einsum('bsi,bic->bsc', res_selected.repeat(self.num_frames,1,1), bev_embed_single)\
                            .view(self.num_frames, bs, learned_latent_query.shape[1],learned_latent_query.shape[2])
        res_latent_query_all = bev_embed_single[:-1] - bev_embed_single[1:]

        learned_latent_query=learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)

        latent_query = self.latent_decoder(
                query=latent_query,
                key=latent_query,
                value=latent_query,
                query_pos=latent_pos,
                key_pos=latent_pos)

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(
            way_point, self.embed_dims, dim=1)

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
                query=way_point,
                key=latent_query,
                value=latent_query,
                query_pos=wp_pos,
                key_pos=latent_pos)
        init_ego_trajs = self.init_ego_fut_decoder(way_point)
        init_ego_trajs = init_ego_trajs.permute(1, 0, 2). view(bs, 
                                                    self.ego_fut_mode, self.fut_ts, 2)
        init_ego_coords = init_ego_trajs.cumsum(dim=2).view(bs, -1, 2)
        init_ego_coords = (init_ego_coords - self.grid_min.to(device).view(1, 1, 2)) / \
                            self.grid_size.to(device).view(1, 1, 2) / 200
        init_wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            init_wp_vector.append(init_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        init_wp_vector = torch.cat(init_wp_vector, dim=1)

        reference_points = init_ego_coords.unsqueeze(2)
        spatial_shapes = torch.tensor([self.bev_w, self.bev_h]).view(1, 2).to(device)
        level_start_index = torch.tensor([0]).to(device)

        res_latent_query=res_latent_query_all[0].permute(1, 0, 2)
        res_latent_query, res_latent_pos = torch.split(
            res_latent_query, self.embed_dims, dim=2)
        res_latent_query = self.action_mln(res_latent_query, init_wp_vector)
        res_latent_query = self.res_latent_decoder(
                query=res_latent_query,
                key=res_latent_query,
                value=res_latent_query,
                query_pos=res_latent_pos,
                key_pos=res_latent_pos)
        
        pred_bev = self.tokenfuser(res_latent_query.permute(1, 0, 2), bev_navi_embed) + bev_navi_embed

        mhst_aux = None
        if self.use_oarwm and osz_mask is not None:
            # Stage-3 MHST: K-hypothesis mixture inside occluded cells.
            #
            # Layout contract (verified against the pipeline):
            #  * pred_bev from the tokenfuser is ALWAYS (B, H*W, C) — the
            #    ResWorld residual path flattens the BEV grid (H=W=100 here,
            #    see the bev_query <-> pos_embd cat) row-major into a
            #    sequence that col_attn consumes via permute(1,0,2).
            #  * MHST is a per-cell convolution head (prior/backbone/experts
            #    are Conv2d), so it operates on the SAME data in 2D grid
            #    form (B, C, H, W). We restore the grid, run the head, and
            #    flatten back before col_attn — a lossless layout change,
            #    not an alternative representation.
            #  * The OSZ mask is (B, 3, 200, 200) (grid_config); the BEV
            #    feature is 100x100, so the mask is downsampled (nearest).
            pb = pred_bev.permute(0, 2, 1).reshape(bs, c, h, w)
            # Gradient contract (ISSUE.md P1-4, completed 2026-08-23):
            # detach the world-model BEV before the MHST / occ heads
            # consume it. P1-4 stopped the additive b_hat branch but left
            # the conv-INPUT path live — the Stage-6 exposure losses
            # (halluc / uncertainty / ground / div / occ_gt) flowed through
            # prior_conv / backbone / experts / sigma_net / occ_head back
            # into pred_bev, pulling the deterministic world model away
            # from serving the planner (occ_frac ~0.67 made this a
            # map-wide distortion, the top suspect for the persistent L2
            # gap vs baseline). With this detach, Stage-6 trains ONLY the
            # MHST/occ heads; the world model is shaped by planning (and
            # depth) losses alone.
            pb = pb.detach()
            mhst_mask = _align_osz_mask(osz_mask, (h, w), device, pb.dtype)
            # (design doc §3.3): col_attn keeps the CLEAN pred_bev — the
            # multi-hypothesis mixture no longer perturbs the shared BEV
            # features consumed by trajectory refinement. The hypotheses
            # reach the planner ONLY via the explicit risk field (Stage 4),
            # so the fused output is computed for its aux (pi/sigma/delta)
            # and then intentionally dropped here.
            mhst_aux = self.mhst(pb, mhst_mask)[1]
            # Extra aux for the Stage-6 losses: un-fused 4D pred_bev, the
            # occlusion occupancy prediction and the aligned osz_eye mask.
            mhst_aux['pred_bev_4d'] = pb
            mhst_aux['mask'] = mhst_mask[:, :1]
            if hasattr(self, 'occ_head'):
                mhst_aux['s_occ'] = self.occ_head(pb)
        elif self.use_oarwm:
            # Fail loudly: MHST without a mask source would silently produce
            # an identity (and leave all its params unused under DDP). The
            # config asserts use_oarwm -> use_osz, so this is a logic error.
            raise ValueError(
                "use_oarwm=True but osz_mask is None — MHST requires a "
                "mask source (use_osz_midas or use_osz_rcsample).")

        risk_field = None
        risk_stats = None
        dyn_raster = None
        if self.use_risk_field and mhst_aux is not None and osz_mask is not None:
            # Stage-4 learnable risk head (V2). All distribution inputs are
            # detached (ISSUE I2): the risk reading never back-shapes the
            # world model / MHST / occ heads.
            phi = (mhst_aux['sigma'] / (1.0 + mhst_aux['sigma'])) * \
                mhst_aux['delta'].norm(dim=2).mean(dim=1, keepdim=True)
            s_occ = mhst_aux.get('s_occ')
            s_occ_in = s_occ.detach() if s_occ is not None else torch.zeros(
                bs, 1, h, w, device=device, dtype=pred_bev.dtype)
            # Geometric proximity 1/(1+d) to the nearest dynamic object,
            # derived from the rasterised detection boxes (V2 Stage 4 input).
            if gt_bboxes is not None:
                dyn_raster = self._rasterise_dynamic(
                    gt_bboxes, img_metas, device=device,
                    forward_margin=self.dyn_forward_margin,
                    vel_thresh=self.dyn_vel_thresh)
                dist_inv = dyn_raster['dist_inv'].to(dtype=pred_bev.dtype)
            else:
                dist_inv = torch.zeros(
                    bs, 1, h, w, device=device, dtype=pred_bev.dtype)
            drivable = None
            if drivable_mask is not None:
                drivable = _align_osz_mask(
                    drivable_mask[:, None, :, :].float(), (h, w),
                    device, pred_bev.dtype)
            cmd_embed = navi_embed[:, 0, :]           # (B, C)
            risk_field, risk_stats = self.risk_head(
                mhst_aux['pred_bev_4d'],              # detached pred_bev (B_t)
                mhst_aux['mask'],                     # strict occlusion M
                phi.detach(),
                s_occ_in,
                dist_inv,
                drivable=drivable,
                cmd_embed=cmd_embed.detach(),
            )

        # Stage-2 risk-gated back-flow (V2): the safety-supervised risk
        # field is written back into the shared planner BEV with a
        # zero-init, task-trained, L1-budgeted per-channel gate. The gate
        # is detached during warmup (injection exactly 0); Proj keeps its
        # default gain so the gate never starves (ISSUE 2.1).
        if self.use_osz and self.osz_inject_mode == 'risk_gated' \
                and risk_field is not None:
            pb4 = pred_bev.permute(0, 2, 1).reshape(bs, c, h, w)
            pb4 = self.osz_fusion(pb4, risk_field, iters=self.iter)
            pred_bev = pb4.reshape(bs, c, h * w).permute(0, 2, 1)

        way_point = self.col_attn(
                query=way_point,
                key=pred_bev.permute(1, 0, 2),
                value=pred_bev.permute(1, 0, 2),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index) 

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2). view(bs, 
                                                      self.ego_fut_mode, self.fut_ts, 2)

        wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            wp_vector.append(outputs_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        wp_vector = torch.cat(wp_vector, dim=1)

        outs = {
            'bev_embed': bev_embed,
            'pred_bev': pred_bev,
            'scene_query': latent_query,
            'wp_vector': wp_vector,
            # 'act_query': act_query,
            # 'act_pos': act_pos,
            'ego_fut_preds': outputs_ego_trajs,
            # 'ego_fut_preds': init_ego_trajs,
            'init_ego_fut_preds': init_ego_trajs,
        }
        if mhst_aux is not None:
            # Stage-3 aux outputs (pi / sigma / delta), consumed by the
            # Stage-6 losses (L_occ_halluc / L_div / L_uncertainty).
            outs['mhst'] = mhst_aux
        if risk_field is not None:
            # Stage-4 risk field [R_exp, R_worst] (B, 2, H, W), consumed
            # by the Stage-5 planner (guard samples R_worst) and by the
            # Stage-2 risk-gated injection.
            outs['risk_field'] = risk_field
        if risk_stats is not None:
            # Smoothed mu / sigma_epis for L_col / L_dyn and the R_safe
            # calibration (collision-positive mu quantile EMA).
            outs['risk_stats'] = risk_stats
        if dyn_raster is not None:
            # Rasterised dynamic occupancy (occ_dyn / s_dyn / dist_inv),
            # shared between the forward (RiskHead proximity input) and the
            # loss (L_col / L_dyn / L_occ_gt targets).
            outs['dyn_raster'] = dyn_raster
        if gt_bev_next is not None:
            # Next-frame exposure ground truth (current ego frame), used by
            # the Stage-6 losses (L_occ_halluc / L_uncertainty).
            outs['gt_bev_next'] = gt_bev_next

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):

        ego_fut_preds = preds_dicts['ego_fut_preds']
        init_ego_fut_preds = preds_dicts['init_ego_fut_preds']
        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_l1_init = self.loss_plan_reg_init(
            init_ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
      
        loss_dict['loss_plan_reg'] = loss_plan_l1
        loss_dict['loss_plan_reg_init'] = loss_plan_l1_init

        # ---- Stage-3/4 related losses (design doc §6.2/6.2b/6.3/6.4) ----
        mhst = preds_dicts.get('mhst')
        if mhst is not None:
            pi = mhst['pi']               # (B, K, H, W)
            sigma = mhst['sigma']         # (B, 1, H, W)
            delta = mhst['delta']         # (B, K, C, H, W)
            mask = mhst['mask']           # (B, 1, H, W) osz_eye
            pb = mhst['pred_bev_4d']      # (B, C, H, W) un-fused pred_bev

            occ_mask = mask > 0
            n_occ = occ_mask.float().sum().clamp(min=1.0)

            # 6.3 L_div: minimise pairwise cosine similarity of the hypothesis
            # residuals (maximise separation). Unit-normalised -> bounded in
            # [-1,1] and gradient-stable (no exploding d2/base ratio).
            # Fixed 2026-08 (ISSUE.md P2-4): (a) computed on OCCLUDED cells
            # only — visible cells never enter the fused output, so pushing
            # their hypotheses apart wasted capacity and had no opposing
            # supervision; (b) clamped at 0 — negatively correlated (already
            # separated) pairs get no reward, the term has a lower bound.
            if self.loss_div_weight > 0 and delta.shape[1] > 1:
                delta_occ = delta * occ_mask.float().unsqueeze(1)  # (B,K,C,H,W)
                flat = delta_occ.flatten(2)                # (B,K,D)
                unit = flat / flat.norm(
                    dim=2, keepdim=True).clamp(min=1e-6)   # (B,K,D)
                cos_pairs = []
                for k1 in range(delta.shape[1]):
                    for k2 in range(k1 + 1, delta.shape[1]):
                        cos_pairs.append(
                            (unit[:, k1] * unit[:, k2]).sum(dim=1).mean())
                loss_dict['loss_div'] = (
                    torch.stack(cos_pairs).mean().clamp(min=0.0)
                    * self.loss_div_weight)

            gt_next = preds_dicts.get('gt_bev_next')   # (B, C, H, W)
            if gt_next is not None:
                # exposure error per cell (mean over C), used by 6.4.
                # P1-4 (fixed 2026-08): the exposure losses train ONLY the
                # MHST head (pi / dB / sigma) — `pb` (the deterministic
                # tokenfuser output) is stop-grad'ed here and `e2` is
                # detached entirely, so the world-model latent residual path
                # is not pulled away from serving the planner, and sigma is
                # calibrated against a FIXED error target (no sigma<->e2
                # mutual-pull loop: |sigma - e2| with a live e2 would drag
                # the error toward sigma instead of calibrating sigma to it).
                b_hat = pb.detach() + (pi.unsqueeze(2) * delta).sum(dim=1)
                # P0-6: cap the calibration TARGET at sigma_max — an outlier
                # exposure error must not drag the |sigma - e2| magnitude
                # (and the sigma it calibrates) to arbitrary values; sigma
                # is separately capped at sigma_max in OcclusionMHSTHead.
                e2 = (b_hat.detach() - gt_next).pow(2).mean(
                    dim=1, keepdim=True).clamp(max=self.mhst_sigma_max)

                # 6.2 L_occ_halluc: negative log mixture likelihood over the
                # occluded cells (EM-style fit of pi / delta / sigma).
                if self.loss_occ_halluc_weight > 0:
                    b_k = pb.detach().unsqueeze(1) + delta   # (B,K,C,H,W)
                    diff2 = (b_k - gt_next.unsqueeze(1)).pow(2).mean(dim=2)
                    var = sigma + 1e-6
                    log_p = -0.5 * (diff2 / var +
                                    torch.log(var) +
                                    math.log(2 * math.pi))
                    log_mix = torch.logsumexp(
                        log_p + torch.log(pi.clamp(min=1e-12)), dim=1)
                    # occ_mask.squeeze(1) -> (B,H,W) matches log_mix
                    loss_dict['loss_occ_halluc'] = (
                        -(log_mix * occ_mask.float().squeeze(1)).sum() / n_occ
                        * self.loss_occ_halluc_weight)

                # 6.4 L_uncertainty: calibrate sigma against exposure error.
                if self.loss_uncertainty_weight > 0:
                    loss_dict['loss_uncertainty'] = (
                        ((sigma - e2).abs() * occ_mask.float()).sum() / n_occ
                        * self.loss_uncertainty_weight)

                # 6.2c L_risk_ground (design doc §4/§6.2): anchor the
                # RAW risk intensity (sigma-weighted hypothesis magnitude,
                # NOT detached — grounding belongs to the distribution-
                # shaping losses, the P0-2 detach contract only guards the
                # risk-field -> planner direction) to the real exposure
                # change e2. The risk field thus becomes an unbiased
                # "future content change" estimate: low along the GT path,
                # so the GT-risk upper bound (§5.1) stays dormant there.
                if self.loss_risk_ground_weight > 0:
                    r_raw = (sigma / (1.0 + sigma)) * delta.norm(
                        dim=2).mean(dim=1, keepdim=True) * mask  # (B,1,H,W)
                    loss_dict['loss_risk_ground'] = (
                        ((r_raw - e2).pow(2) * mask).sum() / n_occ
                        * self.loss_risk_ground_weight)

            # 6.2b L_occ_gt (V2): BCE of the occupancy head against the
            # rasterised DYNAMIC occupancy S_gt (detection boxes with
            # |vel| > dyn_vel_thresh), occluded cells only. pos_weight
            # penalises missed occupancies -> learns the occluded dynamic
            # content instead of the all-zero trivial solution.
            if self.loss_occ_gt_weight > 0 and hasattr(self, 'occ_head') \
                    and 's_occ' in mhst:
                s_occ = mhst['s_occ']     # (B, 1, H, W)
                raster = preds_dicts.get('dyn_raster')
                if raster is None:
                    # use_risk_field=False arm (stage3 ablation): the
                    # forward did not rasterise — do it here once.
                    raster = self._rasterise_dynamic(
                        gt_bboxes_list, img_metas, device=s_occ.device,
                        forward_margin=self.dyn_forward_margin,
                        vel_thresh=self.dyn_vel_thresh)
                occ_gt = raster['occ_dyn'].to(dtype=s_occ.dtype)
                pos = torch.as_tensor(
                    self.loss_occ_gt_pos_weight, dtype=s_occ.dtype,
                    device=s_occ.device)
                loss_dict['loss_occ_gt'] = (
                    F.binary_cross_entropy_with_logits(
                        s_occ, occ_gt, pos_weight=pos, reduction='none')
                    * occ_mask.float()
                ).sum() / n_occ * self.loss_occ_gt_weight

        # ---- Stage-4 safety supervision (V2): L_dyn / L_col + R_safe ----
        # The RiskHead's mu is supervised by the dynamic-occupancy raster
        # S_dyn (forward margin along the heading) and by the sparse
        # collision hard-anchor C_gt (GT ego footprint on dynamic cells).
        # This is the ONLY training signal reaching the risk head (I4: the
        # auxiliary supervision stops there) — the planning losses sample
        # the field detached (below).
        risk_stats = preds_dicts.get('risk_stats')
        raster = preds_dicts.get('dyn_raster')
        if risk_stats is not None and raster is not None:
            mu = risk_stats['mu']                    # (B, 1, H, W) in (0,1)
            s_dyn = raster['s_dyn']                  # (B, 1, H, W) 0/1
            if self.loss_dyn_weight > 0:
                pos = torch.as_tensor(
                    self.loss_dyn_pos_weight, dtype=mu.dtype, device=mu.device)
                loss_dict['loss_dyn'] = (
                    F.binary_cross_entropy(
                        mu, s_dyn.to(dtype=mu.dtype), pos_weight=pos)
                    * self.loss_dyn_weight)
            if self.loss_col_weight > 0:
                c_gt = self._collision_gt(
                    raster['occ_dyn'], ego_fut_gt, fut_masks=ego_fut_masks)
                pos = torch.as_tensor(
                    self.loss_col_pos_weight, dtype=mu.dtype, device=mu.device)
                loss_dict['loss_col'] = (
                    F.binary_cross_entropy(
                        mu, c_gt.to(dtype=mu.dtype), pos_weight=pos)
                    * self.loss_col_weight)
                # Collision-positive cell count of this batch — cached for
                # the [DIAG] line (R_safe calibration fuel).
                self._diag_col_n = int((c_gt > 0.5).float().sum().item())
                # R_safe calibration (design doc Stage 5): EMA over the
                # collision-positive mu quantile. Calibration STARTS at the
                # first batch containing a collision-positive cell —
                # batches without positives are skipped entirely (there is
                # no quantile to calibrate against, and pulling the
                # threshold down on empty batches would keep the guard
                # permanently active). The EMA therefore advances one step
                # per POSITIVE batch: with EMA 0.999 its half-life is
                # ~693 positive batches, so the calibration speed is set by
                # the collision density, not by wall-clock iters. The
                # quantile is all_reduce'ed so every DDP rank's buffer
                # stays identical.
                if self.risk_plan_mode == 'absolute_hinge':
                    pos_cells = mu.detach()[c_gt > 0.5]
                    if pos_cells.numel() > 0:
                        q = torch.quantile(
                            pos_cells, self.risk_safe_quantile)
                        if torch.distributed.is_available() \
                                and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(q)
                            q = q / torch.distributed.get_world_size()
                        # In-place update: assigning would drop the
                        # register_buffer binding (state_dict / .to()).
                        self.risk_safe.mul_(self.risk_safe_ema).add_(
                            q * (1.0 - self.risk_safe_ema))

        # ---- L_gate: L1 bandwidth budget on the injection gate ----
        # Always computed (warmup included) so gate_raw stays in the
        # autograd graph during the frozen period — no DDP
        # unused-parameter failure, no gate drift (tanh(0) = 0).
        if hasattr(self, 'osz_fusion') \
                and getattr(self.osz_fusion, 'mode', 'off') == 'risk_gated':
            g = torch.tanh(self.osz_fusion.gate_raw)
            loss_dict['loss_gate'] = g.abs().sum() * self.loss_gate_weight

        # ---- Stage-5 risk-weighted planning (V2) ----
        # 'absolute_hinge': L_plan_guard = mean_t max(0, R_worst(tau_t) -
        # R_safe) — sparse fallback floor; risk avoidance itself lives in
        # the feature layer (Stage-2 injection). The field is sampled
        # DETACHED so the planning losses cannot reshape the risk content
        # (I2) — the planner only learns to AVOID high-risk regions
        # through the sampling-coordinate (grid) branch.
        # 'gt_relative' (V1 fallback arm): along-path risk upper-bounded
        # by the GT trajectory (+ CVaR ablation term).
        risk_field = preds_dicts.get('risk_field')   # (B, 2, H, W)
        if self.use_risk_plan and risk_field is not None:
            # Risk terms are disabled for the first risk_plan_warmup_epochs
            # epochs (risk head randomly initialised — ISSUE.md P0-1/P1-2).
            # The field is still sampled below so the [DIAG] line keeps
            # reporting r_cmd during warmup.
            risk_active = getattr(self, 'epoch', -1) >= self.risk_plan_warmup_epochs
            # Linear ramp after warmup (0 -> 1 over risk_plan_ramp_epochs):
            # avoids the hard-switch spike in loss_plan_reg observed at the
            # activation epoch (0.53 -> 1.24 in the 2026-08-13 log).
            risk_scale = 1.0
            if risk_active and self.risk_plan_ramp_epochs > 0:
                risk_scale = min(
                    1.0,
                    (getattr(self, 'epoch', self.risk_plan_warmup_epochs)
                     - self.risk_plan_warmup_epochs + 1.0)
                    / self.risk_plan_ramp_epochs)
            # absolute trajectory coords (per-step increments -> cumsum)
            traj_abs = ego_fut_preds.cumsum(dim=2)    # (B, M, T, 2)
            device = traj_abs.device
            gmin = self.grid_min.to(device)
            gsz = self.grid_size.to(device)
            # normalised grid coords (col_attn convention: /grid_size/200)
            nx = (traj_abs[..., 0] - gmin[0]) / (gsz[0] * 200)   # 0-1
            ny = (traj_abs[..., 1] - gmin[1]) / (gsz[1] * 200)
            grid = torch.stack([nx * 2 - 1, ny * 2 - 1], dim=-1)  # (B,M,T,2)
            # GT trajectory (design doc §5): ego_fut_gt is per-step
            # INCREMENTS (matched against the incremental predictions by
            # the L1 above and cumsum'ed in the test-time metric), so
            # cumsum it to absolute coords first — its along-path risk is
            # the "acceptable safety" upper bound (gt_relative mode).
            traj_abs_gt = ego_fut_gt.cumsum(dim=2)      # (B, M, T, 2) absolute
            nx_gt = (traj_abs_gt[..., 0] - gmin[0]) / (gsz[0] * 200)
            ny_gt = (traj_abs_gt[..., 1] - gmin[1]) / (gsz[1] * 200)
            grid_gt = torch.stack(
                [nx_gt * 2 - 1, ny_gt * 2 - 1], dim=-1)          # (B,M,T,2)
            B, M, T, _ = grid.shape
            # Sample both risk channels (channel 0 = R_exp, 1 = R_worst).
            # DETACHED field (I2): the planning loss must not reshape the
            # risk content — only the trajectory (grid branch) moves.
            rf = risk_field.detach()
            sampled = F.grid_sample(
                rf, grid.reshape(B, M * T, 1, 2),
                mode='bilinear', align_corners=True).reshape(B, M * T, 2)
            sampled_gt = F.grid_sample(
                rf, grid_gt.reshape(B, M * T, 1, 2),
                mode='bilinear', align_corners=True).reshape(B, M * T, 2)
            r_worst = sampled[..., 1].reshape(B, M, T)
            r_worst_gt = sampled_gt[..., 1].reshape(B, M, T)
            # command-weighted: only the commanded trajectory is supervised
            cmd_w = ego_fut_cmd  # (B, M) one-hot (squeezed above)
            fut_w = ego_fut_masks.float()             # (B, T) valid flags
            r_cmd = (r_worst * cmd_w.unsqueeze(-1)).sum(dim=1)   # (B, T)
            r_gt_cmd = (r_worst_gt * cmd_w.unsqueeze(-1)).sum(dim=1)  # (B,T)

            if risk_active:
                if self.risk_plan_mode == 'absolute_hinge':
                    # Absolute-threshold safety lower bound (design doc
                    # Stage 5): only steps whose worst-case risk exceeds
                    # the collision-calibrated R_safe are penalised.
                    # Sparse intervention — imitation dominates while the
                    # path stays below the collision-level threshold.
                    loss_dict['loss_plan_guard'] = (
                        ((r_cmd - self.risk_safe).clamp(min=0) * fut_w).sum()
                        / fut_w.sum().clamp(min=1.0)
                        * self.loss_plan_guard_weight * risk_scale)
                else:
                    # gt_relative (V1 fallback arm): GT risk is the
                    # acceptable-safety upper bound; only steps RISKIER
                    # than the human demo are penalised — when pred ~= GT
                    # the term is 0 (zero gradient) and the open-loop L2
                    # regression trains undisturbed.
                    margin = self.risk_plan_margin
                    loss_dict['loss_plan_risk'] = (
                        ((r_cmd - r_gt_cmd - margin).clamp(min=0) * fut_w)
                        .sum() / fut_w.sum().clamp(min=1.0)
                        * self.loss_plan_risk_weight * risk_scale)
                    # CVaR ablation term: tail of the commanded step-risk,
                    # relative to the GT tail. P2-1: fut_w zeroes the
                    # INVALID steps BEFORE topk.
                    if self.loss_plan_cvar_weight > 0:
                        k = max(1, int(math.ceil(T * self.cvar_beta)))
                        r_cmd_valid = r_cmd * fut_w   # (B, T) invalid -> 0
                        r_gt_valid = r_gt_cmd * fut_w  # (B, T) invalid -> 0
                        topk_pred = r_cmd_valid.topk(
                            k, dim=1).values.mean(dim=1)
                        topk_gt = r_gt_valid.topk(
                            k, dim=1).values.mean(dim=1)
                        loss_dict['loss_plan_cvar'] = (
                            (topk_pred - topk_gt - margin).clamp(min=0).mean()
                            * self.loss_plan_cvar_weight * risk_scale)

        # ---- [DIAG] temporary risk-planning diagnostics (remove after tuning) ----
        # Caches a per-iter diagnostic line; DiagLoggerHook prints it at the
        # same cadence as the TextLoggerHook loss output (log_config interval,
        # see projects/mmdet3d_plugin/resworld/hooks/custom_hooks.py). Tracks
        # Stage-5 activation and online-mask quality (ISSUE.md P0-1 / P1-3).
        if self.use_risk_plan and risk_field is not None and mhst is not None:
            msk = mhst['mask'] > 0                   # (B,1,H,W) osz_eye
            occ_frac = msk.float().mean().item()     # occluded-cell fraction
            rf_mean = risk_field.mean().item()       # overall risk mean
            rf_occ = ((risk_field * msk).sum()
                      / msk.float().sum().clamp(min=1.0)).item()
            rc = r_cmd.detach()                      # (B,T) commanded risk
            rgc = r_gt_cmd.detach()                  # (B,T) GT commanded risk
            # V2 diagnostics (ISSUE.md 2.2):
            #   g_l1 = ||g||_1 of the injection gate (must leave 0 after
            #          the warmup; stuck at 0 = gate starvation);
            #   sep_mean/var = sigma_epis spatial mean/var (≈0 = UCB channel
            #          dead, R_worst degraded to R_exp);
            #   guard_act = L_plan_guard activation rate over valid steps
            #          (must be << 1; == 1 means R_safe calibration failed);
            #   risk_safe = current R_safe calibration value (advances only
            #          on batches with collision positives — see the
            #          calibration comment in the L_col block);
            #   col_n = collision-positive cells in this batch (R_safe's
            #          calibration fuel; long runs of 0 = guard dormant).
            g_l1 = 0.0
            if hasattr(self, 'osz_fusion') and \
                    getattr(self.osz_fusion, 'mode', 'off') == 'risk_gated':
                g_l1 = torch.tanh(
                    self.osz_fusion.gate_raw).abs().sum().item()
            sep_mean = 0.0
            sep_var = 0.0
            guard_act = 0.0
            col_n = getattr(self, '_diag_col_n', 0)
            if risk_stats is not None:
                sep = risk_stats['sigma_epis'].detach()
                sep_mean = sep.mean().item()
                sep_var = sep.var().item()
            if risk_active and self.risk_plan_mode == 'absolute_hinge':
                guard_act = (((r_cmd - self.risk_safe) > 0).float() * fut_w
                             ).sum().item() / fut_w.sum().clamp(min=1.0).item()
            # Trajectory-shape diagnostics (over-conservatism probe,
            # 2026-08-16 eval: L2 ~0.57 avg & CR ~1e-4 -> suspected
            # shrunken/slow trajectories; these two stats make the
            # shrinkage visible in the training log):
            #   traj_end = mean end displacement (m) of the commanded traj
            #   traj_step = mean step size (m) of the commanded traj
            tr = (traj_abs * cmd_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            tr = tr * fut_w.unsqueeze(-1)            # (B,T,2) valid steps
            traj_end = tr[:, -1].norm(dim=-1).mean().item()
            traj_step = (tr[:, 1:] - tr[:, :-1]).norm(dim=-1).mean().item()
            self._diag_msg = (
                f"[DIAG] occ_frac={occ_frac:.3f} rf_mean={rf_mean:.5f} "
                f"rf_occ={rf_occ:.5f} r_cmd_mean={rc.mean().item():.5f} "
                f"r_gt_cmd_mean={rgc.mean().item():.5f} "
                f"g_l1={g_l1:.4f} sep_mean={sep_mean:.5f} "
                f"sep_var={sep_var:.5f} guard_act={guard_act:.4f} "
                f"risk_safe={float(self.risk_safe):.4f} col_n={col_n} "
                f"traj_end={traj_end:.2f}m traj_step={traj_step:.2f}m")

        return loss_dict

    def _rasterise_dynamic(self, gt_bboxes_list, img_metas, device,
                           forward_margin=2.0, vel_thresh=0.5):
        """Rasterise DYNAMIC detection boxes (|vel| > ``vel_thresh``) into
        the head's BEV grid (V2, design doc Stage 4/6). Returns

          occ_dyn  : (B, 1, H, W) binary occupancy of dynamic objects
                     (axis-aligned bottom-rectangle approximation, ego
                     frame via ``img_metas[..]['lidar2ego']``);
          s_dyn    : (B, 1, H, W) occ_dyn expanded along each object's
                     heading by ``forward_margin`` (m) — the cells the
                     object is ABOUT to sweep ("ghost-probe" shape), the
                     L_dyn target;
          dist_inv : (B, 1, H, W) 1/(1+d) proximity to the nearest dynamic
                     cell — d approximated by Chebyshev dilation level
                     (3x3 max-pool ring, 0.3 m per level, saturating at
                     ``_DIST_LEVELS``), the RiskHead proximity input.
        """
        B = len(gt_bboxes_list)
        occ = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        s_dyn = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        x_min, y_min = float(self.grid_min[0]), float(self.grid_min[1])
        x_max, y_max = float(self.grid_max[0]), float(self.grid_max[1])
        for b in range(B):
            boxes = gt_bboxes_list[b]
            # Defensive: the test path hands a nested list ([boxes] per
            # sample); the train path hands LiDARInstance3DBoxes directly.
            if isinstance(boxes, (list, tuple)):
                boxes = boxes[0] if len(boxes) else None
            if boxes is None or len(boxes) == 0:
                continue
            c = boxes.corners
            if c.dim() == 2:
                c = c.view(-1, 8, 3)               # defensive: flattened
            corners = c[:, :4, :2]                 # (N,4,2) bottom rect, LiDAR
            corners = corners.to(device=device).float()
            # Dynamic filter on the LiDAR-frame velocity (rotation
            # preserves the norm, so the frame does not matter).
            vel = boxes.tensor[:, 7:9].to(device=device).float()
            dyn = vel.norm(dim=1) > vel_thresh    # (N,)
            if not dyn.any():
                continue
            corners = corners[dyn]
            l2e = None
            if img_metas is not None and len(img_metas) > b:
                l2e = img_metas[b].get('lidar2ego')
            if l2e is not None:
                l2e_t = torch.as_tensor(
                    l2e, dtype=corners.dtype, device=corners.device)
                zero = torch.zeros_like(corners[..., :1])
                one = torch.ones_like(corners[..., :1])
                # homogeneous (x, y, 0, 1): (N,4,4)
                ch = torch.cat([corners[..., :1], corners[..., 1:2],
                                zero, one], dim=-1)
                # explicit einsum: 'ij,bkj->bki' — l2e (4,4) per point
                corners = torch.einsum(
                    'ij,bkj->bki', l2e_t, ch)[..., :2]   # (N,4,2) ego
            xs, ys = corners[..., 0], corners[..., 1]
            # Heading in the EGO frame straight from the corners: front
            # axle midpoint (corners 0,1) minus rear axle midpoint (2,3).
            fwd = (corners[:, :2].mean(dim=1)
                   - corners[:, 2:].mean(dim=1))           # (N,2)
            fwd = fwd / fwd.norm(dim=1, keepdim=True).clamp(min=1e-6)
            dx = fwd[:, 0] * forward_margin
            dy = fwd[:, 1] * forward_margin
            # Y-AXIS CONTRACT (fixed 2026-08, see ISSUE.md P1-5): the BEV
            # feature row 0 is ego-y MINIMUM and the row index grows with
            # ego-y (rcsample create_grid_infos / bevdet gen_grid:
            # row i <-> y = y_min + i * y_step). The previous mapping
            # ``(y_max - ys.max)/range`` was MIRRORED — the occupancy GT
            # landed on the wrong side of the ego.
            def _idx(pts_x, pts_y):
                xi0 = ((pts_x.min(-1).values - x_min) / (x_max - x_min)
                       * self.bev_w).floor().long().clamp(0, self.bev_w - 1)
                xi1 = ((pts_x.max(-1).values - x_min) / (x_max - x_min)
                       * self.bev_w).ceil().long().clamp(1, self.bev_w)
                yi0 = ((pts_y.min(-1).values - y_min) / (y_max - y_min)
                       * self.bev_h).floor().long().clamp(0, self.bev_h - 1)
                yi1 = ((pts_y.max(-1).values - y_min) / (y_max - y_min)
                       * self.bev_h).ceil().long().clamp(1, self.bev_h)
                return xi0, xi1, yi0, yi1
            x0, x1, y0, y1 = _idx(xs, ys)
            for i in range(corners.shape[0]):
                occ[b, 0, y0[i]:y1[i], x0[i]:x1[i]] = 1.0
            # Forward margin: the same rectangle shifted along the heading
            # (bbox union of the original and the shifted rectangle).
            xs_m, ys_m = xs + dx[:, None], ys + dy[:, None]
            xm0, xm1, ym0, ym1 = _idx(xs_m, ys_m)
            x0 = torch.minimum(x0, xm0)
            x1 = torch.maximum(x1, xm1)
            y0 = torch.minimum(y0, ym0)
            y1 = torch.maximum(y1, ym1)
            for i in range(corners.shape[0]):
                s_dyn[b, 0, y0[i]:y1[i], x0[i]:x1[i]] = 1.0
        # 1/(1+d) proximity: Chebyshev dilation rings (3x3 max-pool), 0.3 m
        # per ring (x resolution), saturating at _DIST_LEVELS.
        levels = getattr(self, '_dist_levels', 10)
        cell_m = 0.3
        dist = torch.full_like(occ, levels + 1)
        dist = torch.where(occ > 0.5,
                           torch.zeros_like(occ), dist)
        prev = occ.clone()
        for k in range(1, levels + 1):
            cur = F.max_pool2d(prev, 3, stride=1, padding=1)
            newly = ((cur - prev) > 0.5) & (dist > k)
            dist = torch.where(newly,
                               torch.full_like(dist, k), dist)
            prev = cur
        dist_inv = 1.0 / (1.0 + dist * cell_m)
        return {'occ_dyn': occ, 's_dyn': s_dyn, 'dist_inv': dist_inv}

    def _collision_gt(self, occ_dyn, ego_fut_gt, fut_masks=None,
                      footprint=1.0):
        """Future-collision hard anchor C_gt (V2, design doc Stage 4.4):
        the GT ego trajectory (per-step increments -> absolute, mode 0 —
        all modes are identical after the L1 repeat) is rasterised as an
        ego footprint (``footprint`` m half-extent) and intersected with
        the dynamic occupancy. Cells where the human demo itself would
        enter a dynamic object are the sparse positive anchors of L_col
        and the calibration set of R_safe. ``fut_masks`` (B, T) skips the
        invalid padded steps.
        """
        B = occ_dyn.shape[0]
        traj = ego_fut_gt[:, 0].cumsum(dim=1).detach().cpu()  # (B, T, 2)
        valid = None
        if fut_masks is not None:
            valid = fut_masks.bool().detach().cpu()
        x_min, y_min = float(self.grid_min[0]), float(self.grid_min[1])
        x_max, y_max = float(self.grid_max[0]), float(self.grid_max[1])
        x_res = (x_max - x_min) / self.bev_w
        y_res = (y_max - y_min) / self.bev_h
        rx = max(1, int(math.ceil(footprint / x_res)))
        ry = max(1, int(math.ceil(footprint / y_res)))
        ego_occ = torch.zeros_like(occ_dyn)
        for b in range(B):
            for t in range(traj.shape[1]):
                if valid is not None and not valid[b, t]:
                    continue
                x = float(traj[b, t, 0])
                y = float(traj[b, t, 1])
                if x < x_min or x > x_max or y < y_min or y > y_max:
                    continue
                xi = int(round((x - x_min) / (x_max - x_min) * self.bev_w))
                yi = int(round((y - y_min) / (y_max - y_min) * self.bev_h))
                xi = min(max(xi, 0), self.bev_w - 1)
                yi = min(max(yi, 0), self.bev_h - 1)
                ego_occ[b, 0, max(0, yi - ry):yi + ry + 1,
                        max(0, xi - rx):xi + rx + 1] = 1.0
        return ego_occ * occ_dyn


