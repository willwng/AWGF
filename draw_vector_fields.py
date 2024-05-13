import numpy as np
import matplotlib.pyplot as plt

out_folder = "vector_fields/"


def hamiltonian(q, p) -> float:
    return 0.5 * p ** 2 + 0.5 * q ** 2


def dh_dq(q, p) -> float:
    return q


def dh_dp(q, p) -> float:
    return p


def pretty_plot():
    import scienceplots
    plt.style.use(['science', 'nature'])
    label_size = 18
    plt.xlabel(r"$q$", fontsize=label_size)
    plt.ylabel(r"$p$", fontsize=label_size)
    # Remove ticks
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    plt.grid()


def dissipative_field(q, p, gamma):
    u = np.zeros_like(q)
    v = -gamma * p
    return u, v


def hamiltonian_field(q, p):
    u = dh_dp(q, p)
    v = -dh_dq(q, p)
    return u, v


def fire_field(q, p, gamma):
    u, v = hamiltonian_field(q, p)
    v *= 1 + (gamma * np.linalg.norm(p) / np.linalg.norm(q))
    return u, v


def main():
    gamma = 0.5
    # Meshgrid
    length = 5
    q, p = np.meshgrid(np.linspace(-length, length, 10), np.linspace(-length, length, 10))

    title_fontsize = 18
    # Dissipative field
    fig_size = (5, 5)
    plt.figure(figsize=fig_size)
    plt.title("Dissipative field", fontsize=title_fontsize)
    ud, vd = dissipative_field(q, p, gamma)
    plt.quiver(q, p, ud, vd, color='g')
    pretty_plot()
    plt.savefig(f"{out_folder}dissipative_field.pdf")

    # Hamiltonian field
    plt.figure(figsize=fig_size)
    plt.title("Hamiltonian field", fontsize=title_fontsize)
    uh, vh = hamiltonian_field(q, p)
    plt.quiver(q, p, uh, vh, color='b')
    pretty_plot()
    plt.savefig(f"{out_folder}hamiltonian_field.pdf")

    # Combined field
    plt.figure(figsize=fig_size)
    plt.title("Combined field", fontsize=title_fontsize)
    plt.quiver(q, p, ud + uh, vd + vh, color='r')
    pretty_plot()
    plt.savefig(f"{out_folder}combined_field.pdf")

    # Fire field
    plt.figure(figsize=fig_size)
    plt.title("FIRE Hamiltonian field", fontsize=title_fontsize)
    uf, vf = fire_field(q, p, gamma)
    plt.quiver(q, p, uf, vf, color='y')
    pretty_plot()
    plt.savefig(f"{out_folder}fire_hamiltonian_field.pdf")

    # Combined fire field
    plt.figure(figsize=fig_size)
    plt.title("FIRE field", fontsize=title_fontsize)
    ud, vd = dissipative_field(q, p, gamma)
    plt.quiver(q, p, ud + uf, vd + vf, color='m')
    pretty_plot()
    plt.savefig(f"{out_folder}fire_field.pdf")

    plt.show()

    return


if __name__ == "__main__":
    main()
