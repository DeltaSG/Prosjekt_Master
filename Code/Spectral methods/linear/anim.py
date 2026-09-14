import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# Initial time
t = 0

#fourier plotting
line_fourier, = ax.plot(
    x, U_fourier[0, :],
    label="Fourier"
)
# regular plotting
line_regular, = ax.plot(
    x, U_regular[0, :],
    label="Regular"
)

# analytical solution
if L_operation == "1":
    line_exact, = ax.plot(
        x, exact_L1(u_0, x, t),
        "--",
        label="Analytical"
    )
elif L_operation == "2":
        line_exact, = ax.plot(
            x, exact_L2(x, t, L),
            "--",
            label="Analytical"
        )
elif L_operation == "3":
        line_exact, = ax.plot(
            x, U_fourier[0, :],
            "--",
            label="Analytical"
        )

ax.set_xlim(0, L)

ax.set_ylim(
    U_fourier.min() * 1.1,
    U_fourier.max() * 1.1
)

ax.set_xlabel("x")
ax.set_ylabel("u(t,x)")
ax.legend()

def update(n):
    t = n * k

    line_fourier.set_ydata(U_fourier[n, :])
    line_regular.set_ydata(U_regular[n, :])
    if L_operation == "1":
        line_exact.set_ydata(exact_L1(u_0, x, t))
        ax.set_title(f"u_t + u_x = 0,  t = {t:.2f}")
    elif L_operation == "2":
        line_exact.set_ydata(exact_L2(x, t, L))
        ax.set_title(f"u_t - u_xx = 0,  t = {t:.2f}")
    elif L_operation == "3":
        line_exact.set_ydata(U_fourier[n, :])
        ax.set_title(f"u_t + u_x + u_xxx = 0,  t = {t:.2f}")



    return line_fourier, line_regular, line_exact


animation = FuncAnimation(
    fig,
    update,
    frames=U_fourier.shape[0],
    interval=10,
    blit=False
)

plt.show()