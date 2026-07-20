"""
Visualize the 4 AGGP Ablation Configurations.
Generates a 2×2 diagram showing data flow for each config.

Usage:
    python visualize_configs.py
Output:
    AGGP_ablation_configs.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_config(ax, title, subtitle,
                data_channels, model_input, gate_signal,
                has_gate=True):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.text(5, 6.6, subtitle, ha='center', fontsize=9, style='italic', color='#555')

    speed_color  = '#2196F3'
    accel_color  = '#FF5722'
    model_color  = '#E8EAF6'
    gate_color   = '#FFF3E0'
    output_color = '#E8F5E9'
    unused_color = '#EEEEEE'

    # ── Data input ──
    y_data = 5.5
    label = f"Data: [{', '.join(data_channels)}]"
    rect = FancyBboxPatch((0.5, y_data - 0.3), 3.5, 0.6,
                           boxstyle="round,pad=0.1", fc='#E3F2FD', ec='#1565C0', lw=1.5)
    ax.add_patch(rect)
    ax.text(2.25, y_data, label, ha='center', va='center', fontsize=9, fontweight='bold')

    y_split = y_data - 0.6
    if has_gate:
        ax.annotate('', xy=(1.5, y_split), xytext=(2.25, y_data - 0.3),
                    arrowprops=dict(arrowstyle='->', color=speed_color, lw=2))
        ax.text(0.6, y_split + 0.15, model_input, fontsize=8,
                color=speed_color, fontweight='bold')

        gate_color_line = accel_color if 'accel' in gate_signal.lower() else speed_color
        ax.annotate('', xy=(7.5, 3.85), xytext=(2.25, y_data - 0.3),
                    arrowprops=dict(arrowstyle='->', color=gate_color_line, lw=2, linestyle='--'))
        ax.text(5.2, y_split + 0.15, f"gate: {gate_signal}", fontsize=8,
                color=gate_color_line, fontweight='bold')
    else:
        ax.annotate('', xy=(2.25, 4.15), xytext=(2.25, y_data - 0.3),
                    arrowprops=dict(arrowstyle='->', color=speed_color, lw=2))
        ax.text(0.6, y_split + 0.15, model_input, fontsize=8,
                color=speed_color, fontweight='bold')

    # ── Model backbone ──
    y_model = 3.0
    rect_model = FancyBboxPatch((0.3, y_model - 0.4), 3.8, 1.5,
                                 boxstyle="round,pad=0.15", fc=model_color, ec='#3F51B5', lw=2)
    ax.add_patch(rect_model)
    ax.text(2.25, y_model + 0.7, "GWNet Backbone", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1A237E')
    ax.text(2.25, y_model + 0.2, "TCN → GCN", ha='center', va='center', fontsize=9, color='#333')
    ax.text(2.25, y_model - 0.15, f"start_conv({model_input.split(' ')[0]}ch→H)",
            ha='center', va='center', fontsize=7.5, color='#666')

    # ── Gate (or no-gate) box ──
    if has_gate:
        rect_gate = FancyBboxPatch((5.8, y_model - 0.15), 3.5, 1.0,
                                    boxstyle="round,pad=0.1", fc=gate_color, ec='#E65100', lw=2)
        ax.add_patch(rect_gate)
        ax.text(7.55, y_model + 0.55, "AGGP Gate", ha='center', va='center',
                fontsize=10, fontweight='bold', color='#BF360C')
        ax.text(7.55, y_model + 0.1, "|signal| → Norm → MLP → σ",
                ha='center', va='center', fontsize=8, color='#333')
        ax.annotate('', xy=(4.5, y_model + 0.3), xytext=(5.8, y_model + 0.3),
                    arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
        ax.text(4.7, y_model + 0.55, "×", fontsize=14, fontweight='bold', color='#E65100')
    else:
        rect_gate = FancyBboxPatch((5.8, y_model - 0.15), 3.5, 1.0,
                                    boxstyle="round,pad=0.1", fc=unused_color, ec='#999',
                                    lw=1, linestyle='--')
        ax.add_patch(rect_gate)
        ax.text(7.55, y_model + 0.55, "No Gate", ha='center', va='center',
                fontsize=10, fontweight='bold', color='#999')
        ax.text(7.55, y_model + 0.1, "(disabled)", ha='center', va='center',
                fontsize=8, color='#999')

    # ── Output ──
    y_out = 1.0
    rect_out = FancyBboxPatch((1.5, y_out - 0.3), 3.0, 0.6,
                               boxstyle="round,pad=0.1", fc=output_color, ec='#2E7D32', lw=1.5)
    ax.add_patch(rect_out)
    ax.text(3.0, y_out, "Speed Prediction", ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1B5E20')
    ax.annotate('', xy=(2.25, y_out + 0.3), xytext=(2.25, y_model - 0.4),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("GWNET-AGGP Ablation Study: 4 Configurations",
                 fontsize=16, fontweight='bold', y=0.98)

    draw_config(
        axes[0, 0],
        "Config 1: gwnet_aggp",
        "Full AGGP — 2ch input, gate uses acceleration",
        data_channels=["Speed", "Accel"],
        model_input="2ch [Speed+Accel]",
        gate_signal="Acceleration",
        has_gate=True
    )

    draw_config(
        axes[0, 1],
        "Config 2: gwnet_aggp_gate_speed",
        "1ch input, gate uses speed magnitude",
        data_channels=["Speed"],
        model_input="1ch [Speed]",
        gate_signal="Speed magnitude",
        has_gate=True
    )

    draw_config(
        axes[1, 0],
        "Config 3: gwnet_aggp_gate_accel",
        "Speed → backbone  |  Acceleration → gate only",
        data_channels=["Speed", "Accel"],
        model_input="1ch [Speed only]",
        gate_signal="Acceleration",
        has_gate=True
    )

    draw_config(
        axes[1, 1],
        "Config 4: gwnet (baseline)",
        "Standard GWNet — no AGGP gating",
        data_channels=["Speed", "Accel"],
        model_input="2ch [Speed+Accel]",
        gate_signal="",
        has_gate=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = 'AGGP_ablation_configs.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {out_path}")
