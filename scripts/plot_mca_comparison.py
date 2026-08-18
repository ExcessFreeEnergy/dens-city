import os
os.environ['MPLCONFIGDIR'] = '/home/gauss/code/cdft_sim/dens-city/.mplconfig'
os.makedirs('/home/gauss/code/cdft_sim/dens-city/.mplconfig', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set high-quality styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig.patch.set_facecolor('#0f172a')

for ax in axes.flat:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0', labelsize=10)
    ax.xaxis.label.set_color('#f8fafc')
    ax.yaxis.label.set_color('#f8fafc')
    ax.title.set_color('#f8fafc')
    for spine in ax.spines.values():
        spine.set_color('#475569')
    ax.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')

# -------------------------------------------------------------
# PLOT 1: Liquid-Vapor Binodal of Argon (1st Order vs MCA vs NIST)
# -------------------------------------------------------------
ax1 = axes[0, 0]
T_nist = np.array([85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0, 150.86])
rho_l_nist = np.array([0.02135, 0.02025, 0.01905, 0.01770, 0.01610, 0.01410, 0.01120, 0.00808])
rho_v_nist = np.array([0.00005, 0.00014, 0.00032, 0.00065, 0.00120, 0.00220, 0.00410, 0.00808])

# 1st-Order WCA (Pre-MCA)
T_pre = np.array([85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0, 149.1])
rho_l_pre = np.array([0.01878, 0.01780, 0.01675, 0.01560, 0.01430, 0.01275, 0.01050, 0.00760])
rho_v_pre = np.array([0.00008, 0.00019, 0.00038, 0.00068, 0.00115, 0.00195, 0.00360, 0.00760])

# 2nd-Order MCA (Post-MCA)
T_post = np.array([85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0, 149.4])
rho_l_post = np.array([0.01895, 0.01791, 0.01684, 0.01573, 0.01456, 0.01332, 0.01195, 0.00760])
rho_v_post = np.array([0.00008, 0.00018, 0.00035, 0.00063, 0.00105, 0.00174, 0.00346, 0.00760])

ax1.plot(rho_l_nist, T_nist, 'o-', color='#38bdf8', lw=2.5, label='NIST Experimental Reality', markersize=7)
ax1.plot(rho_v_nist, T_nist, 'o-', color='#38bdf8', lw=2.5, markersize=7)
ax1.plot(rho_l_pre, T_pre, 's--', color='#f43f5e', lw=2.0, label='Pre-MCA (1st-Order WCA)', markersize=6)
ax1.plot(rho_v_pre, T_pre, 's--', color='#f43f5e', lw=2.0, markersize=6)
ax1.plot(rho_l_post, T_post, '^-.', color='#10b981', lw=2.2, label='Post-MCA (2nd-Order Dispersion)', markersize=6)
ax1.plot(rho_v_post, T_post, '^-.', color='#10b981', lw=2.2, markersize=6)

ax1.set_title('A. Pure Argon Coexistence Binodal Envelope', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel(r'Number Density $\rho$ [$\mathrm{\AA}^{-3}$]', fontsize=11)
ax1.set_ylabel(r'Temperature $T$ [K]', fontsize=11)
ax1.legend(loc='lower left', framealpha=0.85, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

# -------------------------------------------------------------
# PLOT 2: Critical Temperature Relative Error Across All Fluids
# -------------------------------------------------------------
ax2 = axes[0, 1]
materials = ['Water\n(H2O)', 'CO2', 'Electrolytes\n(RPM)', 'CO2/H2O\nMixture', 'Nitrogen\n(N2)', 'Methane\n(CH4)', 'Liquid\nCrystals', 'Argon\n(Ar)']
err_pre_tc = [+2.0, -0.01, 0.0, +2.4, +0.01, -2.8, 0.0, -1.17]
err_post_tc = [+2.0, -0.01, 0.0, +2.4, +0.01, -2.8, 0.0, -0.96]

x = np.arange(len(materials))
width = 0.35

rects1 = ax2.bar(x - width/2, err_pre_tc, width, label='Pre-MCA (Tc Err %)', color='#f43f5e', alpha=0.9, edgecolor='#fda4af')
rects2 = ax2.bar(x + width/2, err_post_tc, width, label='Post-MCA (Tc Err %)', color='#10b981', alpha=0.9, edgecolor='#6ee7b7')

ax2.axhline(0, color='#94a3b8', lw=1.0)
ax2.set_title('B. Critical Temperature Relative Error vs Reality', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel('Error vs NIST / Literature [%]', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(materials, fontsize=9.5)
ax2.set_ylim(-4.0, 4.0)
ax2.legend(loc='upper right', framealpha=0.85, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

# -------------------------------------------------------------
# PLOT 3: Absolute Triple-Point / Condensed Liquid Density Errors
# -------------------------------------------------------------
ax3 = axes[1, 0]
fluids_rho = ['Water\n(300K)', 'CO2\n(250K)', 'RPM Salt\n(Reduced)', 'N2\n(Dense)', 'Methane\n(111K)', 'Argon\n(85K)']
rho_err_pre = [1.08, 0.0, 0.0, 0.0, 11.95, 12.24]
rho_err_post = [1.08, 0.0, 0.0, 0.0, 11.95, 11.46]

x3 = np.arange(len(fluids_rho))
b1 = ax3.bar(x3 - width/2, rho_err_pre, width, label='Pre-MCA Density Error (%)', color='#f97316', alpha=0.9)
b2 = ax3.bar(x3 + width/2, rho_err_post, width, label='Post-MCA Density Error (%)', color='#06b6d4', alpha=0.9)

ax3.set_title('C. Absolute Condensed Liquid Density Error Magnitude', fontsize=13, fontweight='bold', pad=12)
ax3.set_ylabel('|Error| vs NIST [%]', fontsize=11)
ax3.set_xticks(x3)
ax3.set_xticklabels(fluids_rho, fontsize=9.5)
ax3.set_ylim(0, 15.0)
ax3.legend(loc='upper right', framealpha=0.85, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

for rect in b1 + b2:
    h = rect.get_height()
    if h > 0:
        ax3.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8.5, color='#e2e8f0', fontweight='bold')

# -------------------------------------------------------------
# PLOT 4: Theoretical Physics Mechanism: Hard-Sphere Compressibility chi_hs(eta)
# -------------------------------------------------------------
ax4 = axes[1, 1]
eta_vals = np.linspace(0.0, 0.55, 200)
chi_hs = ((1.0 - eta_vals)**4) / ((1.0 + 2.0*eta_vals)**2 + (eta_vals**3)*(eta_vals - 4.0))

ax4.plot(eta_vals, chi_hs, color='#a855f7', lw=3.0, label=r'CS Hard-Sphere $\chi_{\mathrm{hs}}(\eta)$')
ax4.axvspan(0.40, 0.50, color='#38bdf8', alpha=0.15, label=r'Liquid Packing Window ($\eta \approx 0.40-0.50$)')
ax4.scatter([0.415], [0.0375], color='#f43f5e', s=120, zorder=5, label=r'Argon Triple Point ($\chi_{\mathrm{hs}} = 0.038$)')

ax4.annotate('Drastic Compressibility Drop\nSuppresses Multi-body Repulsion', 
             xy=(0.415, 0.0375), xytext=(0.15, 0.4),
             arrowprops=dict(facecolor='#38bdf8', shrink=0.08, width=1.5, headwidth=8),
             color='#f8fafc', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#334155', edgecolor='#64748b'))

ax4.set_title(r'D. Physical Mechanism: Reference Compressibility $\chi_{\mathrm{hs}}(\eta)$', fontsize=13, fontweight='bold', pad=12)
ax4.set_xlabel(r'Hard-Sphere Packing Fraction $\eta = \frac{\pi}{6}\rho d^3$', fontsize=11)
ax4.set_ylabel(r'Isothermal Compressibility Factor $\chi_{\mathrm{hs}}$', fontsize=11)
ax4.set_ylim(0, 1.05)
ax4.legend(loc='upper right', framealpha=0.85, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

plt.suptitle('dens-city Engine Verification: First-Order WCA vs. Second-Order MCA Dispersion Across All Pipelines', 
             fontsize=15, fontweight='bold', color='#ffffff', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = '/home/gauss/code/cdft_sim/mca_branch_comparison_chart.png'
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print('Chart saved successfully to:', out_path)
