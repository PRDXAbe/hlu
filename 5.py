import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

fig, ax = plt.subplots(figsize=(11, 5))
names = ["Customer", "MobileApp", "Warehouse", "Drone"]
x = [1, 4, 7, 10]

for p, n in zip(x, names):
    ax.add_patch(Rectangle((p-0.7, 4), 1.4, 0.5, fill=False))
    ax.text(p, 4.25, n, ha="center", va="center")
    ax.plot([p, p], [0.3, 4], "k--", lw=0.7)

steps = [
    (0, 1, "Place Order"),
    (1, 2, "Send Order"),
    (2, 3, "Assign Delivery"),
    (3, 0, "Deliver Package"),
    (0, 1, "Confirm Delivery")
]

for i, (a, b, msg) in enumerate(steps):
    y = 3.5 - i * 0.65
    ax.annotate("", xy=(x[b], y), xytext=(x[a], y),
                arrowprops=dict(arrowstyle="->"))
    ax.text((x[a]+x[b])/2, y+0.1, msg,
            ha="center", fontsize=9)

ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("UML Sequence Diagram")
plt.show()

fig, ax = plt.subplots(figsize=(12, 3))
states = ["Idle", "TakeOff", "Flying",
          "Delivering", "Returning", "Charging"]
x = list(range(1, 12, 2))

for p, s in zip(x, states):
    ax.add_patch(Circle((p, 1.5), 0.7, fill=False))
    ax.text(p, 1.5, s, ha="center", va="center", fontsize=9)

for a, b in zip(x, x[1:]):
    ax.annotate("", xy=(b-0.7, 1.5), xytext=(a+0.7, 1.5),
                arrowprops=dict(arrowstyle="->"))

ax.annotate("", xy=(1, 0.8), xytext=(11, 0.8),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.4"))

ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis("off")
ax.set_title("UML Statechart Diagram")
plt.show()