from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time

def calc_dynamics(x, u, dt):
    th = x[2]
    vx = x[3]
    ax = u[0]
    w = u[1]
    dx = vx*cos(th)*dt
    dy = vx*sin(th)*dt
    dth = w*dt
    dvx = ax*dt
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
        j += self.wgx * (x[0]-x_ref[0])**2
        j += self.wgy * (x[1]-x_ref[1])**2
        j += self.wth * (x[2]-x_ref[2])**2
        j += self.wv * (x[3]-x_ref[3])**2
        j += self.wa1 * u[0]**2
        if self.pre_u is not None:
            j += self.wa2 * (u[0] - self.pre_u[0])**2
        j += self.ww * u[1]**2
        return j



       
class MPC:
    def __init__(self, x0, x_ref_list, rect_list):
        T = 5.0
        N = 20
        dt = T / N
        nx = 4
        nu = 2
        weight = [1.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0]
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
        
        Xk = MX.sym('X0', nx)
        w += [Xk]
        lbw += x0
        ubw += x0
        w0 += [0, 0, 0, 0]
        lam_x0 += [0,0,0,0]
        env = Environment()

        for k in range(N):
            Uk = MX.sym('U_'+str(k), nu)
            Xk_ref = x_ref_list[k]
            x_ref += [Xk_ref]
            w += [Uk]
            lbw += [-1, -1]
            ubw += [1, 1]
            w0 += [0, 0]
            lam_x0 += [0, 0]

            for i, rect in enumerate(rect_list):
                beta = MX.sym('Beta_' + str(i), 4)
                w += [beta]
                lbw += [0, 0, 0, 0]
                ubw += [inf, inf, inf, inf]
                w0 += [0, 0, 0, 0]
                lam_x0 += [0, 0, 0, 0]

                A, b = rect.get(Xk)

                g+=[A[0, 0]*beta[0]+A[1,0]+beta[1]+A[2,0]*beta[2]+A[3,0]*beta[3], A[0, 1]*beta[0]+A[1,1]+beta[1]+A[2,1]*beta[2]+A[3,1]*beta[3]]
                lbg += [0, 0]
                ubg += [0, 0]
                lam_g0 += [0, 0]

                g+= [b[0]*beta[0] + b[1]*beta[1] +b[2]*beta[2] +b[3]*beta[3]]
                lbg += [-inf]
                ubg += [0]
                lam_g0 += [0]

            J = J + cost_function.stage_cost(Xk, Uk, Xk_ref, env)

            dXk = calc_dynamics(Xk, Uk, dt)

            Xk_next = vertcat(Xk[0] + dXk[0],Xk[1] + dXk[1],Xk[2] + dXk[2],Xk[3] + dXk[3])
            Xk1 = MX.sym('X_'+str(k+1), nx)
            w += [Xk1]
            lbw += [-inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf]
            w0 += [0, 0, 0, 0]
            lam_x0 += [0, 0, 0, 0]

            g += [Xk_next-Xk1] 
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

        self.nlp = {'f' : self.J, 'x':self.w, 'g':self.g}
        self.solver = nlpsol('solver', 'ipopt', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'print_time':False, 'ipopt':{'max_iter':10000, 'mu_min':0.1, 'warm_start_init_point':'yes', 'print_level':0, 'print_timing_statistics':'no'}})
    
    def solve(self):
        sol = self.solver(x0 = self.x, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        return sol

class Rect:
    def __init__(self, x, y, th, W, H):
        self.x = x
        self.y = y
        self.W = W
        self.H = H
        self.th = th
        c = np.cos(th)
        s = np.sin(th)

    def calcAb(self, x, y, th):
        c = np.cos(th)
        s = np.sin(th)
        W = self.W
        H = self.H
        A = np.array([[s, -c], [-s, c], [c, s], [-c, -s]])
        print("A", A)
        print("s", s)
        b = np.array([W/2+s*x-c*y, W/2-s*x+c*y, H/2+c*x+s*y, H/2-c*x-s*y]) 
        print("b", b)
        return A, b

    def get(self, xk):
        xk_x = xk[0]
        xk_y = xk[1]
        xk_th = xk[2]
        dx = xk_x - self.x
        dy = xk_y - self.y
        dth = xk_th - self.th
        c = np.cos(self.th)
        s = np.sin(self.th)
        x = dx*c + dy*s
        y = -dx*s + dy*c
        return self.calcAb(x, y, 0)

    def getRectXY(self):
        c = np.cos(self.th)
        s = np.sin(self.th)
        center = np.array([self.x, self.y])
        R = np.array([[c, -s], [s, c]])
        p = np.array([[self.H/2, self.W/2], [-self.H/2, self.W/2], [-self.H/2, -self.W/2], [self.H/2, -self.W/2]])
        x = []
        y = []
        for each_p in p:
            rot_p = R@each_p+center
            x.append(rot_p[0])
            y.append(rot_p[1])
        x.append(x[0])
        y.append(y[0])
        return x, y
        

def main():
    T = 5
    N = 20
    x0 = [0, 1, 0, 0]
    t = [i*T/N for i in range(N)]
    v = 1
    x_ref_list = [[v*t_each, 0, 0, 1] for t_each in t]
    x_ref = [ref[0] for ref in x_ref_list]
    y_ref = [ref[1] for ref in x_ref_list]


    rect_list = [Rect(1, 0, 0.7, 0.5, 0.5)]

    mpc = MPC(x0, x_ref_list, rect_list)
    start = time.time()
    sol = mpc.solve()
    end = time.time()
    print(end-start)
    x = sol['x']
    print(x.shape)

    pos_x = x[0::10]
    pos_y = x[1::10]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pos_x, pos_y, label="opt")
    ax.plot(pos_x, pos_y, "o")
    ax.plot(x_ref, y_ref, label="ref")
    ax.plot(x_ref, y_ref, "o")

    for rect in rect_list:
        x, y = rect.getRectXY()
        ax.plot(x, y, label="obstacle", c="red") 
    ax.legend()
    ax.set_aspect("equal")
    plt.show()



if __name__=="__main__":
    main()

