import math
import matplotlib.pyplot as plt
import numpy as np


# ====================================================
# Pierce / Small-Signal Analytical Codes with Sever
# ====================================================


def cubic_roots(a, b, c):
    """Numerically robust solver using np.roots (eigenvalue method)."""
    poly_coeffs = [1.0, a, b, c]
    roots = np.roots(poly_coeffs)
    return [complex(r) for r in roots]


def get_growing_root(roots):
    """Select growing root with largest Im(δ)."""
    if not roots:
        return None
    return max(roots, key=lambda r: r.imag)


def track_growing_root(roots, previous_root=None):
    """Select growing root with continuity tracking for smooth sweeps."""
    if not roots:
        return None
    if previous_root is None:
        return get_growing_root(roots)
    # Mode tracking: closest root to previous
    selected = min(roots, key=lambda r: abs(r - previous_root))
    return selected


def pierce_roots(Q, C, b, d=0.0):
    """Solve Pierce cubic: δ^3 + QC δ^2 + b δ - C^3 = 0"""
    a = Q * C
    roots_x = cubic_roots(a, b, -C**3)
    roots_delta = [x - 1j * d for x in roots_x]
    return roots_delta


def compute_growth_rate(Q, C, b_sync=0.0):
    """Compute growth rate at synchronism."""
    roots0 = pierce_roots(Q, C, b_sync)
    growing0 = get_growing_root(roots0)
    return growing0.imag


def compute_gains(growth_rate, sever_position, N_total, sever_loss_db):
    """Compute small-signal gains."""
    gain_before_db = 8.686 * growth_rate * sever_position
    gain_after_db = 8.686 * growth_rate * (N_total - sever_position)
    total_gain_with_sever = gain_before_db - sever_loss_db + gain_after_db
    gain_without_sever = 8.686 * growth_rate * N_total
    return gain_before_db, gain_after_db, total_gain_with_sever, gain_without_sever


def heuristic_loop_gain_estimate(gain_before_db, gain_after_db, sever_loss_db,
                                 input_refl_db=40.0, output_refl_db=40.0,
                                 reverse_gain_factor=0.25):
    """
    Heuristic round-trip loop gain estimate (not a full Barkhausen criterion).
    
    This is a simplified model for quick engineering insight.
    Full oscillation prediction requires frequency-dependent phase,
    dispersion, and complex reflection coefficients.
    """
    forward_path = gain_before_db - sever_loss_db - output_refl_db
    reverse_path = (gain_after_db * reverse_gain_factor) - input_refl_db - sever_loss_db
    loop_gain_db = forward_path + reverse_path
    stability_margin_db = -loop_gain_db
    is_stable = loop_gain_db < 0.0
    
    phase_comment = ""
    if abs(stability_margin_db) < 8.0:
        phase_comment = "⚠️  Marginal - full frequency/phase sweep recommended"
    
    return loop_gain_db, stability_margin_db, is_stable, phase_comment


def compute_power_curve(total_gain_with_sever, Pin_dBm_range=None):
    """Compute power transfer curve."""
    if Pin_dBm_range is None:
        Pin_dBm_range = np.linspace(-20, 10, 200)
    Pout_dBm = [power_transfer(p, total_gain_with_sever, 46.0) for p in Pin_dBm_range]
    return Pin_dBm_range, Pout_dBm


def power_transfer(Pin_dBm, Gss_dB=50.0, Psat_dBm=46.0, smoothness=2.0):
    """Empirical soft-saturation model."""
    Pin = 10**((Pin_dBm - 30)/10)
    Pout = (10**(Gss_dB/10) * Pin) / \
           (1 + (10**(Gss_dB/10) * Pin / 10**((Psat_dBm-30)/10))**smoothness)**(1/smoothness)
    return 10 * np.log10(Pout) + 30


# ========================
# Plotting Functions
# ========================

def plot_growth_rate(b_values, growth_rates, b_sync, growth_rate, ax=None):
    """Plot growth-rate curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b_values, growth_rates, 'b-', linewidth=2.5)
    ax.plot(b_sync, growth_rate, 'ro', markersize=8)
    ax.set_title('Growth-Rate Curve\n(zoomed)', fontsize=13)
    ax.set_xlabel('b (velocity parameter)')
    ax.set_ylabel('Im(δ)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.6, 0.6)
    return ax


def plot_phase_velocity(b_values, vp_over_ve, ax=None):
    """Plot normalized phase velocity (standard Pierce approximation)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b_values, vp_over_ve, 'g-', linewidth=2.5)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.6)
    ax.set_title('Normalized Phase Velocity', fontsize=13)
    ax.set_xlabel('b')
    ax.set_ylabel(r'$v_p / v_e$')
    ax.grid(True, alpha=0.3)
    return ax


def plot_cumulative_gain(sever_position, N_total, growth_rate, sever_loss_db, ax=None):
    """Plot cumulative gain profile with sever."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    z = np.linspace(0, N_total, 500)
    gain_profile = 8.686 * growth_rate * z
    mask_after = z > sever_position
    gain_profile[mask_after] -= sever_loss_db
    ax.plot(z, gain_profile, 'purple', linewidth=2.5)
    ax.axvline(sever_position, color='red', linestyle='--', alpha=0.8, label='Sever')
    ax.set_title('Cumulative Gain Profile\n(with sever)', fontsize=13)
    ax.set_xlabel('Normalized Position N')
    ax.set_ylabel('Gain (dB)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_power_transfer(Pin_dBm, Pout_dBm, total_gain_with_sever, ax=None):
    """Plot power transfer curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Pin_dBm, Pout_dBm, 'r-', linewidth=2.5)
    ax.plot(Pin_dBm, np.array(Pin_dBm) + total_gain_with_sever, 'k--', alpha=0.6)
    ax.axhline(46.0, color='green', linestyle='--', alpha=0.7)
    ax.set_title('Power Transfer\n(with sever)', fontsize=13)
    ax.set_xlabel('Pin (dBm)')
    ax.set_ylabel('Pout (dBm)')
    ax.grid(True, alpha=0.3)
    return ax


def create_diagnostic_plots(growth_rates, vp_over_ve, b_values, b_sync, growth_rate,
                           sever_position, N_total, sever_loss_db, total_gain_with_sever):
    """Create all four diagnostic plots in one figure."""
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(1, 4, 1)
    plot_growth_rate(b_values, growth_rates, b_sync, growth_rate, ax1)
    
    ax2 = fig.add_subplot(1, 4, 2)
    plot_phase_velocity(b_values, vp_over_ve, ax2)
    
    ax3 = fig.add_subplot(1, 4, 3)
    plot_cumulative_gain(sever_position, N_total, growth_rate, sever_loss_db, ax3)
    
    Pin_dBm, Pout_dBm = compute_power_curve(total_gain_with_sever)
    ax4 = fig.add_subplot(1, 4, 4)
    plot_power_transfer(Pin_dBm, Pout_dBm, total_gain_with_sever, ax4)
    
    plt.tight_layout()
    plt.savefig('pierce_with_sever_realistic_stability.png', dpi=200, bbox_inches='tight')
    plt.show()


def main():
    # ========================
    # Parameters
    # ========================
    Q = 0.2
    C = 0.04
    N_total = 100
    b_sync = 0.0
    sever_position = 40
    sever_loss_db = 20.0
    return_loss_db = 40.0
    reverse_gain_factor = 0.25  # Empirical: tune based on specific TWT design (0.1-0.4 typical)

    # ========================
    # Computation
    # ========================
    growth_rate = compute_growth_rate(Q, C, b_sync)
    gain_before_db, gain_after_db, total_gain_with_sever, gain_without_sever = compute_gains(
        growth_rate, sever_position, N_total, sever_loss_db
    )
    
    # Heuristic stability estimate
    loop_gain_db, stability_margin_db, is_stable, phase_comment = heuristic_loop_gain_estimate(
        gain_before_db, gain_after_db, sever_loss_db,
        input_refl_db=return_loss_db, output_refl_db=return_loss_db,
        reverse_gain_factor=reverse_gain_factor
    )

    # Growth rate sweep with continuity tracking
    b_values = np.linspace(-0.6, 0.6, 400)
    growth_rates = []
    vp_over_ve = []
    previous_root = None
    for b in b_values:
        roots = pierce_roots(Q, C, b)
        growing = track_growing_root(roots, previous_root)
        growth_rates.append(growing.imag)
        # Standard Pierce normalization for phase velocity
        # β ≈ β_e * (1 + C * Re(δ))  →  v_p / v_e ≈ 1 / (1 + C * Re(δ))
        norm_beta = 1 + C * growing.real
        vp_over_ve.append(1.0 / norm_beta if abs(norm_beta) > 1e-12 else 1.0)
        previous_root = growing

    # ========================
    # Output / Diagnostics
    # ========================
    print("=== Pierce TWT Analysis with Sever ===")
    print(f"Growth rate Im(δ) at b=0   = {growth_rate:.6f}")
    print(f"Gain WITHOUT sever         = {gain_without_sever:.2f} dB")
    print(f"Gain WITH sever            = {total_gain_with_sever:.2f} dB")
    print(f"Loop gain (heuristic)      = {loop_gain_db:.2f} dB")
    print(f"Stability margin           = {stability_margin_db:.2f} dB")
    print("✅ STABLE" if is_stable else "⚠️  OSCILLATION RISK")
    if phase_comment:
        print(phase_comment)

    print("\n=== Root diagnostics ===")
    for b_test in [-0.3, -0.1, 0.0, 0.1, 0.3]:
        roots = pierce_roots(Q, C, b_test)
        growing = get_growing_root(roots)
        im_parts = [f"{r.imag:.2e}" for r in roots]
        print(f"b = {b_test:6.3f} → Im(δ) = {growing.imag:.6f}   (all Im: {im_parts})")

    # ========================
    # Plotting
    # ========================
    create_diagnostic_plots(growth_rates, vp_over_ve, b_values, b_sync, growth_rate,
                           sever_position, N_total, sever_loss_db, total_gain_with_sever)


if __name__ == "__main__":
    main()