import torch
import torch.nn as nn
from model_jit import JiT_models
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import numpy as np
import math
import os
import json

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels, args, noise=None, inverse=False):
        device = labels.device
        bsz = labels.size(0)
        if noise is None:
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        else:
            z = self.noise_scale * noise
        if inverse:
            start = 1.0
            end = 0.0
        else:
            start = 0.0
            end = 1.0
        timesteps = torch.linspace(start, end, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif 'euler_ours' in self.method:
            stepper = self._euler_step_ours

        elif self.method == "heun":
            stepper = self._heun_step

        elif 'heun_ours' in self.method:
            stepper = self._heun_step_ours
        else:
            raise NotImplementedError

        # ode
        for i in tqdm(range(self.steps - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if 'ours' in self.method:
                z = stepper(z, t, t_next, labels, args=args, iter=args.iter, perturb=args.perturb_scale)
            else:
                z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    # @torch.no_grad()
    def _forward_sample(self, z, t, labels, grad=False):

        # if self.cfg_scale > 0:
        #     # conditional
        #     x_cond = self.net(z, t.flatten(), labels)
        #     v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # if self.cfg_scale != 1.0:
        #     # unconditional
        #     x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        #     v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        batched_z_input = torch.cat([z, z], dim=0)
        batched_t_input = torch.cat([t.flatten(), t.flatten()], dim=0)
        batched_labels_input = torch.cat([labels, torch.full_like(labels, self.num_classes)], dim=0)

        if grad:
            x = self.net(batched_z_input, batched_t_input, batched_labels_input)
        else:
            with torch.no_grad():
                x = self.net(batched_z_input, batched_t_input, batched_labels_input)

        x_cond, x_uncond = torch.chunk(x, 2, dim=0)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)


        if self.cfg_scale == 1.0:
            return v_cond
        elif self.cfg_scale == 0.0:
            return v_uncond
        else:
            return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)


    def get_scheduled_value(self, total, cur, schedule_type):
        if schedule_type == 'constant':
            return total

        elif schedule_type == 'linear': # 1 - (t/T)
            return total * (1. - cur / total)

        elif schedule_type == 'cosine':
            return total * 0.5 * (1. + math.cos(math.pi * cur / total))

        elif schedule_type == 'sqrt':
           return total * math.sqrt(1. - cur / total)

        elif schedule_type == 'concave':
           return (1. - cur / total) ** 2

        elif schedule_type == 'convex':
           return 1. - (cur / total) ** 2

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")


class DenoiserCustom(Denoiser):
    LOCAL_METRIC_METHODS = {
        "heun_mc_directional": "directional",
        "heun_mc_acceleration": "acceleration",
        "heun_mc_spectral": "spectral",
        "heun_mc_lipschitz": "lipschitz",
        "heun_lip_min": "lipschitz",
    }

    def __init__(self, args):
        super().__init__(args)
        # Raw per-sample/per-timestep records for local-metric MC runs.
        self._local_metric_trace_batches = []
        self._active_local_metric_trace = None
        self._spectral_direction = None
        self._local_metric_timepoints = 0

    @torch.no_grad()
    def generate(self, labels, args, noise=None, inverse=False, sample_ids=None):
        device = labels.device
        bsz = labels.size(0)
        if noise is None:
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        else:
            z = self.noise_scale * noise
        if inverse:
            start = 1.0
            end = 0.0
        else:
            start = 0.0
            end = 1.0
        timesteps = torch.linspace(start, end, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step

        elif self.method == "heun":
            stepper = self._heun_step

        elif self.method in {
            "heun_consist_mc",
            "heun_randdir",
            "heun_mc_accept_all",
            "heun_mc_reuse_all",
            "heun_fds_gd_nograd",
            "heun_fds_gd",
            "heun_consistency_gd",
            "heun_lip_min",
            "heun_lip_min_3candidates",
            "heun_lip_min_gd",
            *self.LOCAL_METRIC_METHODS,
        }:
            stepper = self._heun_step

        elif 'ours' not in self.method:
            raise NotImplementedError

        improved = None
        delta = None
        experiment_methods = {
            "heun_consist_mc", "heun_randdir", "heun_mc_accept_all",
            "heun_mc_reuse_all", "heun_fds_gd_nograd", "heun_fds_gd",
            "heun_consistency_gd", "heun_lip_min", "heun_lip_min_3candidates",
            "heun_lip_min_gd",
            *self.LOCAL_METRIC_METHODS,
        }
        batch_index = getattr(self, "_experiment_batch_index", 0)
        fixed_direction = None
        if self.method in experiment_methods:
            self._experiment_batch_index = batch_index + 1
            if self.method in {"heun_randdir", "heun_mc_reuse_all"}:
                gen = torch.Generator(device=device).manual_seed(args.seed_delta + batch_index)
                fixed_direction = torch.randn(z.shape, generator=gen, device=device, dtype=z.dtype)
                if self.method == "heun_randdir":
                    norm = fixed_direction.flatten(1).norm(dim=1).clamp_min(1e-12)
                    fixed_direction = fixed_direction * (math.sqrt(float(np.prod(z.shape[1:]))) / norm).view(-1, *([1] * (z.ndim - 1)))
        metric_name = self.LOCAL_METRIC_METHODS.get(self.method)
        if metric_name is not None or self.method == "heun_lip_min_gd":
            if sample_ids is None:
                sample_ids = torch.arange(bsz, device=device, dtype=torch.long)
            else:
                sample_ids = sample_ids.to(device=device, dtype=torch.long)
            if sample_ids.shape != (bsz,):
                raise ValueError(
                    f"sample_ids must have shape {(bsz,)}, got {tuple(sample_ids.shape)}"
                )
            if metric_name is not None:
                self._active_local_metric_trace = {
                    "sample_ids": sample_ids.detach().cpu(),
                    "t": [],
                    "dt": [],
                    "base_metric": [],
                    "candidate_metric": [],
                    "selected_metric": [],
                    "accepted": [],
                    "power_iterations": [],
                    "metric": metric_name,
                }
            # One right direction per sample is carried through this
            # trajectory and updated only from the selected candidate.
            self._spectral_direction = None
            self._local_metric_timepoints = 0
        # ode
        for i in tqdm(range(self.steps - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if self.method in experiment_methods:
                z = self._experimental_heun_step(z, t, t_next, labels, args, batch_index, i, fixed_direction)
                continue
            if 'ours' in self.method:
                # z = stepper(z, t, t_next, labels, args=args, iter=args.iter, perturb=args.perturb_scale)
                v_func_kwargs = {
                    'z': z,
                    't': t,
                    't_next': t_next,
                    'labels': labels}

                iter = args.iter
                if t.mean().item() > args.stop_t:
                    iter = 0

                # hard-coded as euler ours -> TODO: generalize
                v_func = None
                update_func = None
                if 'euler' in self.method:
                    v_func = self._euler_get_v_pred
                    update_func = self._euler_update
                elif 'heun' in self.method:
                    v_func = self._heun_get_v_pred
                    update_func = self._heun_update
                else:
                    raise NotImplementedError
                delta_schedule = self.get_scheduled_value(1., t.mean().item(), schedule_type=args.perturb_schedule)
                best_z, best_v_pred, improved, delta = self.divergence_stepper(v_func,
                                            v_func_kwargs,
                                            x_key='z',
                                            t_key='t',
                                            stop_t=args.stop_t,
                                            num_updates=iter,
                                            delta_scale=args.perturb_scale * delta_schedule, #/ iter if iter > 0 else args.perturb_scale * delta_schedule, # currently hard-coded
                                            # delta_scale=args.perturb_scale* 0.5 * (1. + math.cos(math.pi * t.mean().item())),
                                            # delta_scheduler=lambda n: 1,
                                            delta_scheduler=lambda n: 2**(-n),
                                            seed_delta=args.seed_delta,
                                            seed_eps=args.seed_eps,
                                            improved=improved,
                                            delta=delta,
                                            num_delta=args.num_delta
                )
                # z = best_z + (t_next - t) * best_v_pred # z_next
                z = update_func(best_z, t, t_next, best_v_pred)

            else:
                z = stepper(z, t, t_next, labels)
        # last step euler - (default setup of jit)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        if self._active_local_metric_trace is not None:
            trace = self._active_local_metric_trace
            for key in (
                "t",
                "dt",
                "base_metric",
                "candidate_metric",
                "selected_metric",
                "accepted",
                "power_iterations",
            ):
                trace[key] = torch.stack(trace[key], dim=0)
            self._local_metric_trace_batches.append(trace)
            self._active_local_metric_trace = None
        return z

    # @torch.no_grad()
    def _euler_get_v_pred(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)

        return v_pred

    def _euler_update(self, z, t, t_next, v_pred):
        z = z + (t_next - t) * v_pred # z_next
        return z

    def _heun_get_v_pred(self, z, t, t_next, labels, grad=False):
        v_pred_t = self._forward_sample(z, t, labels, grad=grad)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, grad=grad)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)

        return v_pred

    def _heun_update(self, z, t, t_next, v_pred):
        z_next = z + (t_next - t) * v_pred
        return z_next

    def _metric_forward(self, z, t, labels):
        """Evaluate CFG velocity with autograd enabled inside generate()."""
        with torch.enable_grad():
            return self._forward_sample(z, t, labels, grad=True)

    def _velocity_jvp(self, z, t, labels, tangent, create_graph=False):
        """Return v and J_v @ tangent without materializing J_v."""
        def velocity_fn(x):
            return self._metric_forward(x, t, labels)

        with torch.enable_grad():
            if hasattr(torch, "func") and hasattr(torch.func, "jvp"):
                velocity, jv = torch.func.jvp(
                    velocity_fn, (z,), (tangent,), strict=False
                )
            else:
                velocity, jv = torch.autograd.functional.jvp(
                    velocity_fn, z, tangent, create_graph=create_graph, strict=False
                )
        if create_graph:
            return velocity, jv
        return velocity.detach(), jv.detach()

    def _velocity_vjp(self, z, t, labels, cotangent, create_graph=False):
        """Return v and J_v^T @ cotangent for a batched velocity field."""
        with torch.enable_grad():
            x = z.detach().requires_grad_(True)
            velocity = self._metric_forward(x, t, labels)
            scalar = (velocity * cotangent).flatten(1).sum(1)
            vjp = torch.autograd.grad(
                scalar.sum(),
                x,
                create_graph=create_graph,
                retain_graph=False,
            )[0]
        if create_graph:
            return velocity, vjp
        return velocity.detach(), vjp.detach()

    def _directional_metric(self, z, t, labels):
        velocity = self._metric_forward(z, t, labels).detach()
        velocity_norm = velocity.flatten(1).norm(dim=1).clamp_min(1e-8)
        shape = (velocity_norm.shape[0],) + (1,) * (z.ndim - 1)
        unit_velocity = velocity / velocity_norm.view(shape)
        _, jv = self._velocity_jvp(z, t, labels, unit_velocity)
        return (unit_velocity * jv).flatten(1).sum(1), None

    @staticmethod
    def _normalize_sample_vectors(v):
        norm = v.flatten(1).norm(dim=1).clamp_min(1e-8)
        shape = (norm.shape[0],) + (1,) * (v.ndim - 1)
        return v / norm.view(shape)

    def _spectral_metric(self, z, t, labels, warm_direction, num_iterations):
        """Warm-start power iteration for lambda_max((J + J^T) / 2)."""
        direction = self._normalize_sample_vectors(warm_direction)
        score = None
        for _ in range(num_iterations):
            _, jv = self._velocity_jvp(z, t, labels, direction)
            _, vjp = self._velocity_vjp(z, t, labels, direction)
            symmetric_product = 0.5 * (jv + vjp)
            score = (direction * symmetric_product).flatten(1).sum(1)
            direction = self._normalize_sample_vectors(symmetric_product)
        return score, direction.detach()

    def _lipschitz_metric(self, z, t, labels, warm_direction, num_iterations):
        """Power iteration for ||J_v||_2 via J_v^T J_v."""
        direction = self._normalize_sample_vectors(warm_direction)
        lipschitz = None
        for _ in range(num_iterations):
            # a = J_v @ u
            _, jv = self._velocity_jvp(z, t, labels, direction)
            # b = J_v^T @ a = J_v^T J_v @ u
            _, jt_jv = self._velocity_vjp(z, t, labels, jv)
            lipschitz = jv.flatten(1).norm(dim=1)
            direction = self._normalize_sample_vectors(jt_jv)
        return lipschitz, direction.detach()

    def _initial_metric_direction(
        self, z, args, batch_index, step_index
    ):
        generator = torch.Generator(device=z.device).manual_seed(
            int(args.seed_eps) + batch_index * 100003 + step_index
        )
        return torch.randn(
            z.shape,
            generator=generator,
            device=z.device,
            dtype=z.dtype,
        )

    def _metric_power_iterations(self, metric_name):
        if metric_name in {"spectral", "lipschitz"}:
            return 5 if self._local_metric_timepoints == 0 else 1
        return 0

    def _lip_min_gd_step(self, z, t, t_next, labels, args, batch_index, step_index):
        """Descend a truncated power-iteration estimate of ||J_v||_2.

        The power-iteration direction is treated as fixed while differentiating
        the final JVP. This requires second derivatives of the velocity model,
        but avoids differentiating through the whole warm-start iteration.
        """
        power_iterations = self._metric_power_iterations("lipschitz")
        direction = self._spectral_direction
        if direction is None:
            direction = self._initial_metric_direction(
                z, args, batch_index, step_index
            )

        with torch.no_grad():
            _, direction = self._lipschitz_metric(
                z, t, labels, direction, power_iterations
            )

        # Higher-order derivatives require a single dtype through the graph;
        # the outer generation path may be running under bfloat16 autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.enable_grad():
                current_z = z.detach().float().requires_grad_(True)
                _, jv = self._velocity_jvp(
                    current_z,
                    t,
                    labels,
                    direction.detach().float(),
                    create_graph=True,
                )
                lipschitz = jv.flatten(1).norm(dim=1)
                score_grad = torch.autograd.grad(
                    lipschitz.sum(),
                    current_z,
                    create_graph=False,
                    retain_graph=False,
                )[0]

        grad_norm = score_grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        grad_shape = (grad_norm.shape[0],) + (1,) * (score_grad.ndim - 1)
        schedule = self.get_scheduled_value(
            1.0, float(t.mean().item()), args.perturb_schedule
        )
        step = (
            args.perturb_scale
            * schedule
            * math.sqrt(float(np.prod(z.shape[1:])))
            * score_grad.detach()
            / grad_norm.view(grad_shape)
        )
        updated_z = current_z.detach() - step
        self._spectral_direction = direction.detach()
        self._local_metric_timepoints += 1
        return self._heun_step(updated_z, t, t_next, labels)

    @staticmethod
    def _normalized_gradient_step(current_z, score_grad, step_scale):
        """Take a direction-normalized step with a Gaussian-shell radius."""
        dimension = float(np.prod(current_z.shape[1:]))
        grad_norm = score_grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        shape = (grad_norm.shape[0],) + (1,) * (score_grad.ndim - 1)
        step = step_scale * math.sqrt(dimension) * score_grad / grad_norm.view(shape)
        return current_z.detach() - step.detach()

    def _fds_gd_step(self, z, t, t_next, labels, args, batch_index, step_index):
        """Descend the Hutchinson divergence of the Heun velocity field."""
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.enable_grad():
                current_z = z.detach().float().requires_grad_(True)
                generator = torch.Generator(device=z.device).manual_seed(
                    int(args.seed_eps) + batch_index * 100003 + step_index
                )
                eps = torch.randn(
                    current_z.shape,
                    generator=generator,
                    device=z.device,
                    dtype=current_z.dtype,
                )
                velocity = self._heun_get_v_pred(
                    current_z, t, t_next, labels, grad=True
                )
                projection = (velocity * eps).flatten(1).sum(1)
                jteps = torch.autograd.grad(
                    projection.sum(),
                    current_z,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                divergence = (jteps * eps).flatten(1).sum(1) / np.prod(
                    current_z.shape[1:]
                )
                divergence_grad = torch.autograd.grad(
                    divergence.sum(), current_z, create_graph=False
                )[0]

        schedule = self.get_scheduled_value(
            1.0, float(t.mean().item()), args.perturb_schedule
        )
        updated_z = self._normalized_gradient_step(
            current_z, divergence_grad, args.perturb_scale * schedule
        )
        return self._heun_step(updated_z, t, t_next, labels)

    def _consistency_score(self, z, t, t_next, labels, grad=False):
        """Squared Heun forward/backward round-trip error."""
        forward_velocity = self._heun_get_v_pred(
            z, t, t_next, labels, grad=grad
        )
        forward_z = z + (t_next - t) * forward_velocity
        backward_velocity = self._heun_get_v_pred(
            forward_z, t_next, t, labels, grad=grad
        )
        backward_z = forward_z + (t - t_next) * backward_velocity
        return (backward_z - z).flatten(1).pow(2).mean(1)

    def _consistency_gd_step(self, z, t, t_next, labels, args):
        """Descend the full Heun forward/backward round-trip error."""
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.enable_grad():
                current_z = z.detach().float().requires_grad_(True)
                score = self._consistency_score(
                    current_z, t, t_next, labels, grad=True
                )
                score_grad = torch.autograd.grad(
                    score.sum(), current_z, create_graph=False
                )[0]

        schedule = self.get_scheduled_value(
            1.0, float(t.mean().item()), args.perturb_schedule
        )
        updated_z = self._normalized_gradient_step(
            current_z, score_grad, args.perturb_scale * schedule
        )
        return self._heun_step(updated_z, t, t_next, labels)

    @torch.no_grad()
    def _lip_min_mc_step(
        self, z, t, t_next, labels, args, batch_index, step_index, num_candidates
    ):
        """Select Gaussian candidates by the finite-difference Lipschitz score."""
        schedule = self.get_scheduled_value(
            1.0, float(t.mean().item()), args.perturb_schedule
        )
        scale = args.perturb_scale * schedule
        generator = torch.Generator(device=z.device).manual_seed(
            int(args.seed_delta) + batch_index * 100003 + step_index
        )
        base_velocity = self._heun_get_v_pred(z, t, t_next, labels)
        best_score = None
        best_candidate = None
        shape = (z.shape[0],) + (1,) * (z.ndim - 1)
        for _ in range(num_candidates):
            direction = torch.randn(
                z.shape,
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )
            displacement = scale * direction
            candidate = z + displacement
            candidate_velocity = self._heun_get_v_pred(
                candidate, t, t_next, labels
            )
            score = (
                (candidate_velocity - base_velocity).flatten(1).norm(dim=1)
                / displacement.flatten(1).norm(dim=1).clamp_min(1e-12)
            )
            if best_score is None:
                best_score = score
                best_candidate = candidate
                continue
            choose = score < best_score
            best_score = torch.where(choose, score, best_score)
            best_candidate = torch.where(
                choose.view(shape), candidate, best_candidate
            )
        return self._heun_step(best_candidate, t, t_next, labels)

    def _acceleration_metric(self, z, t, t_next, labels):
        with torch.no_grad():
            velocity = self._forward_sample(z, t, labels)
            # Use the same Heun predictor as the sampler, not an Euler preview.
            next_z = self._heun_step(z, t, t_next, labels)
            next_velocity = self._forward_sample(next_z, t_next, labels)
        dt = (t_next - t).flatten()[0].abs().clamp_min(1e-8)
        velocity_norm = velocity.flatten(1).norm(dim=1)
        acceleration = (next_velocity - velocity).flatten(1).norm(dim=1)
        return acceleration / (dt * velocity_norm + 1e-8), None

    def _compute_local_metric(
        self,
        metric_name,
        z,
        t,
        t_next,
        labels,
        args,
        batch_index,
        step_index,
        warm_direction=None,
        num_iterations=1,
    ):
        if metric_name == "directional":
            return self._directional_metric(z, t, labels)
        if metric_name == "acceleration":
            return self._acceleration_metric(z, t, t_next, labels)
        if metric_name == "spectral":
            return self._spectral_metric(
                z, t, labels, warm_direction, num_iterations
            )
        if metric_name == "lipschitz":
            return self._lipschitz_metric(
                z, t, labels, warm_direction, num_iterations
            )
        raise ValueError(f"Unknown local metric: {metric_name}")

    def _record_local_metric(
        self,
        t,
        t_next,
        base_metric,
        candidate_metric,
        accepted,
        power_iterations,
    ):
        if self._active_local_metric_trace is None:
            return
        trace = self._active_local_metric_trace
        trace["t"].append(t.mean().detach().cpu())
        trace["dt"].append((t_next - t).flatten()[0].abs().detach().cpu())
        trace["base_metric"].append(base_metric.detach().float().cpu())
        trace["candidate_metric"].append(candidate_metric.detach().float().cpu())
        selected_metric = torch.where(accepted, candidate_metric, base_metric)
        trace["selected_metric"].append(selected_metric.detach().float().cpu())
        trace["accepted"].append(accepted.detach().cpu())
        trace["power_iterations"].append(
            torch.tensor(power_iterations, dtype=torch.int64)
        )

    @staticmethod
    def _summarize_values(values):
        values = values.float().flatten()
        quantile_levels = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95],
            dtype=values.dtype,
            device=values.device,
        )
        quantiles = torch.quantile(values, quantile_levels)
        return {
            "count": int(values.numel()),
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "std": float(values.std(unbiased=False).item()),
            "p5": float(quantiles[0].item()),
            "p25": float(quantiles[1].item()),
            "p50": float(quantiles[2].item()),
            "p75": float(quantiles[3].item()),
            "p95": float(quantiles[4].item()),
            "max": float(values.max().item()),
        }

    def _build_local_metric_summary(self):
        selected_values = [
            batch["selected_metric"].reshape(-1)
            for batch in self._local_metric_trace_batches
        ]
        if not selected_values:
            return None
        selected_values = torch.cat(selected_values, dim=0)

        trajectory_mean = []
        trajectory_max = []
        trajectory_integral = []
        trajectory_ids = []
        for batch in self._local_metric_trace_batches:
            selected = batch["selected_metric"].float()
            dt = batch["dt"].float().view(-1, 1)
            trajectory_mean.append(selected.mean(dim=0))
            trajectory_max.append(selected.max(dim=0).values)
            trajectory_integral.append((selected * dt).sum(dim=0))
            trajectory_ids.append(batch["sample_ids"].long())

        trajectory = {
            "sample_ids": torch.cat(trajectory_ids).tolist(),
            "mean": torch.cat(trajectory_mean).tolist(),
            "max": torch.cat(trajectory_max).tolist(),
            "integral": torch.cat(trajectory_integral).tolist(),
        }
        return {
            "pointwise_selected": self._summarize_values(selected_values),
            "trajectory_mean": self._summarize_values(
                torch.as_tensor(trajectory["mean"])
            ),
            "trajectory_max": self._summarize_values(
                torch.as_tensor(trajectory["max"])
            ),
            "trajectory_integral": self._summarize_values(
                torch.as_tensor(trajectory["integral"])
            ),
            "trajectory": trajectory,
        }

    @torch.no_grad()
    def save_local_metric_trace(self, save_folder, rank):
        if not self._local_metric_trace_batches:
            return None
        path = os.path.join(
            save_folder, f"local_metric_trace_rank{int(rank):03d}.pt"
        )
        summary = self._build_local_metric_summary()
        torch.save(
            {
                "format_version": 1,
                "method": self.method,
                "metric": self.LOCAL_METRIC_METHODS[self.method],
                "batches": self._local_metric_trace_batches,
                "summary": summary,
            },
            path,
        )
        summary_path = os.path.join(
            save_folder, f"local_metric_summary_rank{int(rank):03d}.json"
        )
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        return path

    @torch.no_grad()
    def divergence_stepper(self, v_func,
                           v_func_kwargs,
                           x_key='z',
                           t_key='t',
                           stop_t=1.0,
                           num_updates=1,
                           num_delta=1,
                           num_eps=1,
                           delta_scale=1,
                           delta_scheduler=lambda t: 2 ** (-t),
                           seed_delta=None,
                           seed_eps=None,
                           delta=None,
                           improved=None,
                           sequential_hutchinson=True,
                           ):
        assert stop_t >= 0.0 and stop_t <= 1.0

        t = v_func_kwargs[t_key]
        if isinstance(t, torch.Tensor):
            t_flat = t.reshape(-1)
            if not torch.all(t_flat == t_flat[0]).item():
                raise AssertionError(
                    "All timesteps in the batch must be the same for divergence_stepper."
                )
            t = t_flat[0].item()

        if num_updates <= 0 or t > stop_t:
            return v_func_kwargs[x_key], v_func(**v_func_kwargs), improved, delta

        z = v_func_kwargs[x_key]
        B = z.shape[0]
        D = np.prod(z.shape[1:])  # C * H * W

        delta_generator = None
        eps_generator = None

        if seed_delta is not None:
            delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + int(t * 1000))
        if seed_eps is not None:
            eps_generator = torch.Generator(device=z.device).manual_seed(seed_eps)
        sync_eps_with_delta = num_eps == 1 and seed_eps == seed_delta

        for update_idx in range(num_updates):
            # compute divergence and find the best perturbation
            for delta_idx in range(num_delta+1):

                if delta is None or improved is None:
                    assert improved is None and t == 0.0 and delta_idx <= 1
                    delta = torch.randn(z.shape, generator=delta_generator, device=z.device)
                elif update_idx > 0:
                    temp_delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + update_idx)
                    temp_delta = torch.randn(z.shape, generator=temp_delta_generator, device=z.device)
                    pass
                elif delta_idx > 0:
                    new_delta = torch.randn(z.shape, generator=delta_generator, device=z.device)
                    delta = torch.where(
                        improved.reshape(-1, *([1]*(z.ndim-1))), # hard-coded shape
                        delta, # True
                        new_delta # False
                    )
                # no update delta when delta_idx=0
                assert seed_delta != seed_eps, "Is a Biased Estimator"

                if sync_eps_with_delta and delta_idx != 0:
                    eps = delta.detach()
                    raise NotImplementedError # not using anymore!
                else:
                    eps = torch.randn(z.shape, generator=eps_generator, device=z.device)

                if delta_idx == 0:
                    perturbed_z = z
                elif update_idx == 0:
                    perturbed_z = z + delta_scale * delta_scheduler(update_idx) * delta # TODO: clarify
                else:
                    perturbed_z = z + delta_scale * delta_scheduler(update_idx) * temp_delta # TODO: clarify
                with torch.enable_grad():
                    perturbed_z = perturbed_z.detach().requires_grad_(True)
                    v_func_kwargs[x_key] = perturbed_z

                    v_pred = v_func(**v_func_kwargs)  # [B, C, H, W]
                    v_pred_eps = (v_pred * eps).flatten(1).sum(1)  # [B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [B]
                        inputs=perturbed_z,                      # [B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [B]
                        create_graph=False,
                        retain_graph=False,
                    )[0].detach()  # [B, C, H, W]
                    divergence = (grad_v * eps).flatten(1).sum(1) / D  # [B]

                threshold = - (1 / (1 - t))

                if delta_idx == 0:
                    best_divergence = divergence.detach()
                    best_v_pred = v_pred.detach()
                    best_perturbed_z = perturbed_z.detach()
                elif update_idx == 0:
                    improved = (divergence < best_divergence) & (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    best_perturbed_z = torch.where(
                        improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )
                else:
                    temp_improved = (divergence < best_divergence) & (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(temp_improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        temp_improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    best_perturbed_z = torch.where(
                        temp_improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )

            # update iteration-wise
            z = best_perturbed_z # update z
            v_pred = best_v_pred
        return best_perturbed_z, best_v_pred, improved, delta


    def _experimental_heun_step(self, z, t, t_next, labels, args, batch_index, step_index, fixed_direction):
        t_value = float(t.mean().item())
        stop_t = 0.1 if self.method == "heun_fds_gd_nograd" else args.stop_t
        if t_value > stop_t:
            return self._heun_step(z, t, t_next, labels)
        schedule = self.get_scheduled_value(1.0, t_value, args.perturb_schedule)
        if self.method == "heun_fds_gd_nograd":
            with torch.enable_grad():
                x = z.detach().requires_grad_(True)
                gen = torch.Generator(device=z.device).manual_seed(int(args.seed_eps) + batch_index * 100003 + step_index)
                eps = torch.randn(x.shape, generator=gen, device=z.device, dtype=z.dtype)
                v = self._heun_get_v_pred(x, t, t_next, labels, grad=True)
                proj = (v * eps).flatten(1).sum(1)
                jteps = torch.autograd.grad(proj.sum(), x, create_graph=True, retain_graph=True)[0]
                div = (jteps * eps).flatten(1).sum(1) / np.prod(x.shape[1:])
                grad = torch.autograd.grad(div.sum(), x)[0]
            accept = div.detach() >= -1.0 / (1.0 - t_value)
            shape = (z.shape[0],) + (1,) * (z.ndim - 1)
            z = torch.where(accept.view(shape), x.detach() - args.gd_grad_scale * schedule * grad.detach(), x.detach())
            return self._heun_step(z, t, t_next, labels)
        if self.method == "heun_fds_gd":
            return self._fds_gd_step(
                z, t, t_next, labels, args, batch_index, step_index
            )
        if self.method == "heun_consistency_gd":
            return self._consistency_gd_step(z, t, t_next, labels, args)
        if self.method == "heun_lip_min_3candidates":
            return self._lip_min_mc_step(
                z, t, t_next, labels, args, batch_index, step_index, 3
            )
        if self.method == "heun_lip_min_gd":
            return self._lip_min_gd_step(
                z, t, t_next, labels, args, batch_index, step_index
            )
        scale = args.perturb_scale * schedule
        local_metric_method = self.LOCAL_METRIC_METHODS.get(self.method)
        if self.method in {"heun_consist_mc", "heun_mc_accept_all"} or local_metric_method is not None:
            gen = torch.Generator(device=z.device).manual_seed(args.seed_delta + batch_index * 100003 + step_index)
            direction = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
            candidate = z + scale * direction
            if self.method == "heun_mc_accept_all":
                return self._heun_step(candidate, t, t_next, labels)
            if local_metric_method is not None:
                power_iterations = self._metric_power_iterations(local_metric_method)
                if local_metric_method in {"spectral", "lipschitz"}:
                    previous_direction = self._spectral_direction
                    if previous_direction is None:
                        previous_direction = self._initial_metric_direction(
                            z, args, batch_index, step_index
                        )
                else:
                    previous_direction = None
                base_metric, base_direction = self._compute_local_metric(
                    local_metric_method,
                    z,
                    t,
                    t_next,
                    labels,
                    args,
                    batch_index,
                    step_index,
                    previous_direction,
                    power_iterations,
                )
                candidate_metric, candidate_direction = self._compute_local_metric(
                    local_metric_method,
                    candidate,
                    t,
                    t_next,
                    labels,
                    args,
                    batch_index,
                    step_index,
                    previous_direction,
                    power_iterations,
                )
                choose = candidate_metric < base_metric
                shape = (z.shape[0],) + (1,) * (z.ndim - 1)
                selected = torch.where(choose.view(shape), candidate, z)
                if local_metric_method in {"spectral", "lipschitz"}:
                    self._spectral_direction = torch.where(
                        choose.view(shape),
                        candidate_direction,
                        base_direction,
                    ).detach()
                self._record_local_metric(
                    t,
                    t_next,
                    base_metric,
                    candidate_metric,
                    choose,
                    power_iterations,
                )
                self._local_metric_timepoints += 1
                return self._heun_step(selected, t, t_next, labels)
            base_fwd = self._heun_step(z, t, t_next, labels)
            base_back = self._heun_step(base_fwd, t_next, t, labels)
            cand_fwd = self._heun_step(candidate, t, t_next, labels)
            cand_back = self._heun_step(cand_fwd, t_next, t, labels)
            base_err = (base_back - z).flatten(1).pow(2).mean(1)
            cand_err = (cand_back - candidate).flatten(1).pow(2).mean(1)
            choose = cand_err < base_err
            shape = (z.shape[0],) + (1,) * (z.ndim - 1)
            selected = torch.where(choose.view(shape), candidate, z)
            return self._heun_step(selected, t, t_next, labels)
        if self.method in {"heun_randdir", "heun_mc_reuse_all"}:
            return self._heun_step(z + scale * fixed_direction, t, t_next, labels)
        raise NotImplementedError(self.method)
