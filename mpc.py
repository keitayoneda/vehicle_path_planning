from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time


def calc_dynamics(x, u, dt):
    th = x[2]
    vx = x[3]
    ax = u[0]
    w = u[1]
    dx = vx * cos(th) * dt
    dy = vx * sin(th) * dt
    dth = w * dt
    dvx = ax * dt
    return dx, dy, dth, dvx


class Environment:
    pass


class CostFunction:
    def __init__(self, weight):
        self.wgx = weight[0]
        self.wgy = weight[1]
        self.wth = weight[2]
        self.wv = weight[3]
        self.wa1 = weight[4]
        self.wa2 = weight[5]
        self.ww = weight[6]
        self.wc = weight[7]
        self.wp = weight[8]
        self.pre_u = None

    def lane_distance(self, x, env=None):
        return 0

    def potential_func(self, x, env=None):
        return 0

    def stage_cost(self, x, u, x_ref, env=None):
        j = 0
        j += self.wgx * (x[0] - x_ref[0]) ** 2
        j += self.wgy * (x[1] - x_ref[1]) ** 2
        j += self.wth * (x[2] - x_ref[2]) ** 2
        j += self.wv * (x[3] - x_ref[3]) ** 2
        j += self.wa1 * u[0] ** 2
        if self.pre_u is not None:
            j += self.wa2 * (u[0] - self.pre_u[0]) ** 2
        j += self.ww * u[1] ** 2
        return j


class MPC:
    def __init__(self, T, N, x0, x_ref_list, ellipses):
        dt = T / N
        nx = 4
        nu = 2
        weight = [1.0, 3.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        cost_function = CostFunction(weight)

        w = []
        x_ref = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        lam_x0 = []
        lam_g0 = []

        Xk = MX.sym("X0", nx)
        w += [Xk]
        lbw += x0
        ubw += x0
        w0 += [0, 0, 0, 0]
        lam_x0 += [0, 0, 0, 0]
        env = Environment()

        for k in range(N):
            Uk = MX.sym("U_" + str(k), nu)
            Xk_ref = x_ref_list[k]
            x_ref += [Xk_ref]
            w += [Uk]
            lbw += [-1, -1]
            ubw += [1, 1]
            w0 += [0, 0]
            lam_x0 += [0, 0]
            J = J + cost_function.stage_cost(Xk, Uk, Xk_ref, env)

            for i, ellipse in enumerate(ellipses):
                g += [ellipse.getEq(Xk)]
                lbg += [-inf]
                ubg += [0.0]
                lam_g0 += [0.0]
                J = J + ellipse.getPotential(Xk)
            dXk = calc_dynamics(Xk, Uk, dt)

            Xk_next = vertcat(
                Xk[0] + dXk[0], Xk[1] + dXk[1], Xk[2] + dXk[2], Xk[3] + dXk[3]
            )
            Xk1 = MX.sym("X_" + str(k + 1), nx)
            w += [Xk1]
            lbw += [-inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf]
            w0 += [0, 0, 0, 0]
            lam_x0 += [0, 0, 0, 0]

            g += [Xk_next - Xk1]
            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]
            lam_g0 += [0, 0, 0, 0]
            Xk = Xk1

        self.J = J
        self.w = vertcat(*w)
        self.g = vertcat(*g)
        self.x = w0
        self.lam_x = lam_x0
        self.lam_g = lam_g0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

        self.nlp = {"f": self.J, "x": self.w, "g": self.g}
        self.solver = nlpsol(
            "S",
            "ipopt",
            self.nlp,
        )

    def solve(self):
        sol = self.solver(
            x0=self.x,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            lam_x0=self.lam_x,
            lam_g0=self.lam_g,
        )
        return sol


class Rect:
    def __init__(self, x, y, th, W, H):
        self.x = x
        self.y = y
        self.W = W
        self.H = H
        self.th = th

    def calcAb(self, _x, _y, _th):
        th = self.th
        x = self.x
        y = self.y
        c = np.cos(th)
        s = np.sin(th)
        W = self.W
        H = self.H
        A = np.array([[s, -c], [-s, c], [c, s], [-c, -s]])
        b = np.array(
            [
                W / 2 + s * x - c * y,
                W / 2 - s * x + c * y,
                H / 2 + c * x + s * y,
                H / 2 - c * x - s * y,
            ]
        )
        print(f"A:{A}")
        print(f"b:{b}")
        return A, b

    def get(self, xk):
        xk_x = xk[0]
        xk_y = xk[1]
        x = self.x - xk_x
        y = self.y - xk_y
        return self.calcAb(x, y, self.th)

    def getRectXY(self):
        c = np.cos(self.th)
        s = np.sin(self.th)
        center = np.array([self.x, self.y])
        R = np.array([[c, -s], [s, c]])
        p = np.array(
            [
                [self.H / 2, self.W / 2],
                [-self.H / 2, self.W / 2],
                [-self.H / 2, -self.W / 2],
                [self.H / 2, -self.W / 2],
            ]
        )
        x = []
        y = []
        for each_p in p:
            rot_p = R @ each_p + center
            x.append(rot_p[0])
            y.append(rot_p[1])
        x.append(x[0])
        y.append(y[0])
        return x, y


class nEllipse:
    def __init__(self, x, y, W, H, th, n=2):
        self.x = x
        self.y = y
        self.W = W
        self.H = H
        self.th = th
        self.n = n

    def getEq(self, Xk):
        x = Xk[0]
        y = Xk[1]
        c = np.cos(self.th)
        s = np.sin(self.th)
        X_rot = c * (x - self.x) + s * (y - self.y)
        Y_rot = -s * (x - self.x) + c * (y - self.y)
        return 1 - (X_rot / self.W) ** self.n - (Y_rot / self.H) ** self.n

    def getEq2(self, Xk):
        x = Xk[0]
        y = Xk[1]
        c = np.cos(self.th)
        s = np.sin(self.th)
        X_rot = c * (x - self.x) + s * (y - self.y)
        Y_rot = -s * (x - self.x) + c * (y - self.y)
        return (X_rot / self.W) ** 2 + (Y_rot / self.H) ** 2

    def getPotential(self, Xk):
        return 1 / (self.getEq2(Xk) - 1.0 + 1e-6)

    def getXYForPlot(self):
        th = np.linspace(0, np.pi * 0.5, 30)
        x_tmp = self.W * (np.cos(th) ** (2 / self.n))
        y_tmp = self.H * (np.sin(th) ** (2 / self.n))
        x_tmp = np.append(x_tmp, -x_tmp[::-1])
        x_tmp = np.append(x_tmp, -x_tmp)

        y_tmp = np.append(y_tmp, y_tmp[::-1])
        y_tmp = np.append(y_tmp, -y_tmp)

        x = x_tmp * np.cos(self.th) - y_tmp * np.sin(self.th) + self.x
        y = x_tmp * np.sin(self.th) + y_tmp * np.cos(self.th) + self.y

        return x, y


def main():
    T = 5
    N = 60
    x0 = [0, 0 + 0.1, 0, 0]
    t = [i * T / N for i in range(N)]
    v = 1
    x_ref_list = [[v * t_each, 0, 0, 1] for t_each in t]
    x_ref = [ref[0] for ref in x_ref_list]
    y_ref = [ref[1] for ref in x_ref_list]

    # rect_list = [Rect(1, 0, 0.7, 0.5, 1), Rect(4, 0, 0., 0.5, 0.5), Rect(6, 1.5, 0.0, 0.5, 0.5), Rect(9, 0, 0., 0.7, 0.5)]
    ellipses = [nEllipse(2.5, 0.0, 1.0, 0.5, 0, 4)]
    # rect_list = []

    mpc = MPC(T, N, x0, x_ref_list, ellipses)
    start = time.time()
    sol = mpc.solve()
    end = time.time()
    print(end - start)
    x = sol["x"]
    # print(sol)

    nu = 2
    nx = 4
    n_ele = nu + nx
    pos_x = x[0::n_ele]
    pos_y = x[1::n_ele]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pos_x, pos_y, label="opt")
    # ax.plot(pos_x, pos_y, "o")
    ax.plot(x_ref, y_ref, label="ref")
    # ax.plot(x_ref, y_ref, "o")
    # ax.plot(beta_1, label = "beta1")
    # ax.plot(beta_2, label = "beta2")
    # ax.plot(beta_3, label = "beta3")
    # ax.plot(beta_4, label = "beta4")
    # ax.legend()
    # plt.show()

    for ellipse in ellipses:
        x, y = ellipse.getXYForPlot()
        ax.plot(x, y, label="obstacle", c="red")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
