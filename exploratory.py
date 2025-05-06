import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator

class RelativePermeability:

    def __init__(self, infos):

        # Residuals
        self.swc = infos['swc']
        self.sgr = infos['sgr']
        self.sor = infos['sor']

        # Exponents
        self.nw = infos['nw']
        self.ng = infos['ng']
        self.no = infos['no']

        # End-points
        self.krw0 = infos['krw0']
        self.krg0 = infos['krg0']
        self.kro0 = infos['kro0']
    
    def krw(self, Sw):
        return self.krw0 * np.power(np.divide(Sw - self.swc, 1 - self.swc - self.sgr - self.sor), self.nw)

    def krg(self, Sg):
        return self.krg0 * np.power(np.divide(Sg - self.sgr, 1 - self.swc - self.sgr - self.sor), self.ng)

    def kro(self, So):
        return self.kro0 * np.power(np.divide(So - self.sor, 1 - self.swc - self.sgr - self.sor), self.no)

if __name__ == '__main__':
    
    ## CORE INFO
    core = {
        'k': 1.98, # [D]
        'phi': 0.22 # [-]
    }
    core['k'] *= 9.869233e-13 # convert from [D] to [m2]

    ## RELATIVE PERMEABILITY MODEL
    infos = {
        'swc':0.135,
        'sgr':0.2,
        'sor':0.1,
        'nw':2.46,
        'ng':1.3,
        'no':2.0,
        'krw0':0.713,
        'krg0':0.94,
        'kro0':0.5
    }
    relPerm = RelativePermeability(infos)

    ## FLUIDS   - T = 35 [C]
    muw = 0.70e-3       # [Pa.s]       
    muo = 5.00e-3       # [Pa.s]
    mug = 2.07e-5       # [Pa.s]
    sigma_wg = 0.03     # [N/m]

    ## FOAM-QUALITY SCAN
    ut = 3.0 # [ft/day]
    ut *= 3.52778e-6 # Convert from [ft/day] to [m/s]

    fgMuapp = np.array([
        [0.31351869606903165, 0.6133682830930537],
        [0.34036433365292423, 0.6408912188728701],
        [0.36625119846596355, 0.6566186107470511],
        [0.3969319271332694, 0.673656618610747],
        [0.42569511025886864, 0.6998689384010484],
        [0.45349952061361454, 0.7169069462647444],
        [0.48226270373921376, 0.7444298820445608],
        [0.5129434324065196, 0.7614678899082568],
        [0.5417066155321189, 0.7771952817824377],
        [0.5752636625119847, 0.7994757536041939],
        [0.6049856184084371, 0.8217562254259501],
        [0.6337488015340363, 0.8492791612057666],
        [0.7996164908916586, 0.8768020969855831],
        [0.8341323106423777, 0.8217562254259501],
        [0.9031639501438159, 0.783748361730013],
        [0.9079578139980824, 0.7391874180865006],
        [0.9137104506232022, 0.6959370904325033],
        [0.9204218600191754, 0.6513761467889907],
        [0.9252157238734419, 0.618610747051114],
        [0.9338446788111218, 0.5792922673656618],
        [0.9367209971236816, 0.5360419397116645],
        [0.9482262703739214, 0.49803407601572736],
        [0.9511025886864811, 0.45347313237221487]
    ])

    ## PLOT 
    plt.plot(*zip(*fgMuapp), '+')
    plt.xlabel(r'$f_g$ [-]', fontsize=14)
    plt.ylabel(r'$\mu_{app}$ [Pa.s]', fontsize=14)
    plt.grid()
    # plt.show()
    plt.savefig('muApp_fg.png', dpi=300)
    plt.close()

    ## GET grad(p)
    gradP = - fgMuapp[:,1] * ut / core['k'] # [Pa/m]
    # gradP *= 4.419e-5 # [psi/ft]

    ## PLOT
    plt.plot(fgMuapp[:,0], gradP*4.419e-5, '+') # convert from [Pa/m] to [psi/ft]
    plt.xlabel(r'$f_g$ [-]', fontsize=14)
    plt.ylabel(r'$\nabla p$ [psi/ft]', fontsize=14)
    # plt.ylabel(r'$\nabla p$ [psi/ft]', fontsize=14)
    plt.grid()
    # plt.show()
    plt.savefig('gradP_fg.png', dpi=300)
    plt.close()

    ## EVALUATING PHASE VELOCITIES
    fg = fgMuapp[:,0]
    ug = fg * ut
    uw = (ut - ug) * (4/5)
    uo = (1/4) * uw

    # ug *= 283465 # convert from [m/s] to [ft/day]
    # uw *= 283465 # convert from [m/s] to [ft/day] 
    # uo *= 283465 # convert from [m/s] to [ft/day]
    ut_calc = uw + ug + uo 

    ## SCATTER PLOT
    uwug = np.array([
        [0.08793709559635210, 0.8910891089108910,  93],
        [0.18826995907137890, 0.8712871287128712, 181],
        [0.24227026476232538, 0.8118811881188117, 237],
        [0.29632661379345480, 0.8613861386138622, 198],
        [0.74143810607475850, 0.8613861386138622, 196],
        [1.25086187864069440, 0.8415841584158423, 158],
        [1.74486354295807010, 0.8514851485148514, 144],
        [1.24914491449144950, 2.5049504950495054, 217],
        [0.74739907953059460, 2.4455445544554450, 200],
        [0.48970670651970870, 1.6633663366336630, 204],
        [0.49525499719783317, 2.4455445544554450, 204],
        [0.49049640813137940, 3.1980198019801980, 206],
        [0.25338722551500437, 2.4158415841584160, 194],
        [0.08932289455360631, 3.5841584158415842, 104],
        [0.22595655791994296, 4.1089108910891090, 201],
        [0.19853098517398915, 5.8118811881188130, 201],
        [0.08809503591868623, 6.1980198019801980, 107]
    ])
    
    scatter = plt.scatter(uw*283465, ug*283465, c=np.abs(gradP)*4.419e-5, cmap='viridis', edgecolor='k')
    scatter = plt.scatter(uwug[:,0], uwug[:,1], c=uwug[:,2], cmap='viridis', edgecolor='k')
    plt.plot([uw[0]*283465, uw[-1]*283465], [ug[0]*283465, ug[-1]*283465], 'k--')
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'$\nabla p$ [psi/ft]')
    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('uwug_gradP.png', dpi=300)
    plt.close()

    ## GRIDDATA
    uw_grid = np.linspace(min(uwug[:,0]), max(uwug[:,0]), 200)
    ug_grid = np.linspace(min(uwug[:,1]), max(uwug[:,1]), 200)
    uw_mesh, ug_mesh = np.meshgrid(uw_grid, ug_grid)
    grid_points = np.column_stack((uwug[:,0], uwug[:,1]))
    grad_P_grid = griddata(grid_points, uwug[:,2], (uw_mesh, ug_mesh), method='linear')
    contourf = plt.contourf(uw_mesh, ug_mesh, grad_P_grid, levels=20, cmap='jet')
    plt.scatter(uwug[:,0], uwug[:,1], c=uwug[:,2], cmap='jet', edgecolor='k')
    scatter = plt.scatter(uw*283465, ug*283465, c=np.abs(gradP)*4.419e-5, cmap='jet', edgecolor='k')
    plt.plot([uw[0]*283465, uw[-1]*283465], [ug[0]*283465, ug[-1]*283465], 'k--')
    cbar = plt.colorbar(contourf)
    cbar.set_label('Pressure Gradient')
    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0,1.8])
    plt.ylim([0,7])
    # plt.show()
    plt.savefig('uwug_gradP_griddata.png', dpi=300)
    plt.close()



    ## DESIRED UT
    def interpolate_gradP(ut_des, uw_des):
        ug_des = ut_des - (5/4) * uw_des
        uo_des = (1/4) * uw_des
        interp_func = RegularGridInterpolator((ug_grid, uw_grid), grad_P_grid)
        points = np.column_stack((ug_des, uw_des))
        grad_P_des = interp_func(points)
        return ug_des, grad_P_des, uo_des
    
    ut_des = 3.0 # [ft/day]
    uw_des = np.linspace(0.1,1.7,20)

    # ut_des = 2.3 # [ft/day]
    # uw_des = np.linspace(0.1,1.1,10)

    # ut_des = 4.0 # [ft/day]
    # uw_des = np.linspace(0.1,1.3,20)

    ug_des, grad_P_des, uo_des = interpolate_gradP(ut_des, uw_des)
    contourf = plt.contourf(uw_mesh, ug_mesh, grad_P_grid, levels=20, cmap='jet')
    plt.scatter(uwug[:,0], uwug[:,1], c=uwug[:,2], cmap='jet', edgecolor='k')
    scatter = plt.scatter(uw_des, ug_des, c=grad_P_des, cmap='jet', edgecolor='k')
    plt.plot([uw_des[0], uw_des[-1]], [ug_des[0], ug_des[-1]], 'k--')
    cbar = plt.colorbar(contourf)
    cbar.set_label('Pressure Gradient')
    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0,1.8])
    plt.ylim([0,7])
    # plt.show()
    plt.savefig('uwug_gradP_desired.png', dpi=300)
    plt.close()

    ## DIFFERENT UTs
    ut_des_1 = 2.3 # [ft/day]
    uw_des_1 = np.linspace(0.1,1.1,10)
    ug_des_1, grad_P_des_1, uo_des_1 = interpolate_gradP(ut_des_1, uw_des_1)
    fg_des_1 = ug_des_1 / (uw_des_1 + ug_des_1 + uo_des_1)

    ut_des_2 = 3.0 # [ft/day]
    uw_des_2 = np.linspace(0.1,1.7,20)
    ug_des_2, grad_P_des_2, uo_des_2 = interpolate_gradP(ut_des_2, uw_des_2)
    fg_des_2 = ug_des_2 / (uw_des_2 + ug_des_2 + uo_des_2)

    ut_des_3 = 4.0 # [ft/day]
    uw_des_3 = np.linspace(0.1,1.3,20)
    ug_des_3, grad_P_des_3, uo_des_3 = interpolate_gradP(ut_des_3, uw_des_3)
    fg_des_3 = ug_des_3 / (uw_des_3 + ug_des_3 + uo_des_3)

    plt.plot(fg_des_1, grad_P_des_1, label=f'$u_t = {ut_des_1}$')
    plt.plot(fg_des_2, grad_P_des_2, label=f'$u_t = {ut_des_2}$')
    plt.plot(fg_des_3, grad_P_des_3, label=f'$u_t = {ut_des_3}$')
    plt.xlabel(r'$f_g$ [-]', fontsize=14)
    plt.ylabel(r'$\nabla p$ [psi/ft]', fontsize=14)
    # plt.ylabel(r'$\nabla p$ [psi/ft]', fontsize=14)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig('differentUts.png', dpi=300)
    plt.close()

    contourf = plt.contourf(uw_mesh, ug_mesh, grad_P_grid, levels=20, cmap='jet')
    plt.scatter(uwug[:,0], uwug[:,1], c=uwug[:,2], cmap='jet', edgecolor='k')
    scatter = plt.scatter(uw_des_1, ug_des_1, c=grad_P_des_1, cmap='jet', edgecolor='k')
    plt.plot([uw_des_1[0], uw_des_1[-1]], [ug_des_1[0], ug_des_1[-1]], 'k--')
    scatter = plt.scatter(uw_des_2, ug_des_2, c=grad_P_des_2, cmap='jet', edgecolor='k')
    plt.plot([uw_des_2[0], uw_des_2[-1]], [ug_des_2[0], ug_des_2[-1]], 'k--')
    scatter = plt.scatter(uw_des_3, ug_des_3, c=grad_P_des_3, cmap='jet', edgecolor='k')
    plt.plot([uw_des_3[0], uw_des_3[-1]], [ug_des_3[0], ug_des_3[-1]], 'k--')
    cbar = plt.colorbar(contourf)
    cbar.set_label('Pressure Gradient')
    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0,1.8])
    plt.ylim([0,7])
    # plt.show()
    plt.savefig('three_uwug_gradP_desired.png', dpi=300)
    plt.close()


    ## COMPARED DESIRED VS TANG
    fg_des = ug_des / (uw_des + ug_des + uo_des)
    plt.plot(fg_des, grad_P_des, c='b', label='Interpolated from experimental')
    plt.plot(fgMuapp[:,0], -gradP*4.419e-5,'ko', label='(Tang, J., 2018)') # convert from [Pa/m] to [psi/ft]
    plt.xlabel(r'$f_g$ [-]', fontsize=14)
    plt.ylabel(r'$\nabla p$ [psi/ft]', fontsize=14)
    # plt.ylabel(r'$\nabla p$ [psi/ft]', fontsize=14)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig('comparison_desiredVStang.png', dpi=300)
    plt.close()
    
    ## PLOT VELOCITIES FG
    plt.plot(fg,uw*283465,c='b',label=r'$u_w$')
    plt.plot(fg,ug*283465,c='g',label=r'$u_g$')
    plt.plot(fg,uo*283465,c='r',label=r'$u_o$')
    plt.plot(fg,ut_calc*283465,c='k',label=f'$u_t$')
    plt.xlabel(r'$f_g$ [-]')
    plt.ylabel(r'$u$ [ft/day]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('velocities_fg.png', dpi=300)
    plt.close()

    ## FIND Sw and So
    Sw = []
    for i in range(len(fg)):
        func = lambda Sw: np.abs(relPerm.krw(Sw) + uw[i] * muw / (core['k'] * gradP[i]))
        res = minimize_scalar(func, bounds=(0, 1))
        Sw.append(res.x)
    Sw = np.array(Sw)
    # print(Sw)

    So = []
    for i in range(len(fg)):
        func = lambda So: np.abs(relPerm.kro(So) + uo[i] * muo / (core['k'] * gradP[i]))
        res = minimize_scalar(func, bounds=(0, 1))
        So.append(res.x)
    So = np.array(So)
    # print(So)

    ## PLOT
    Sg = 1.0 - Sw - So
    plt.plot(fg,Sw,c='b',label=r'$S_w$')
    plt.plot(fg,Sg,c='g',label=r'$S_g$')
    plt.plot(fg,So,c='r',label=r'$S_o$')
    plt.xlabel(r'$f_g$ [-]')
    plt.ylabel(r'phase saturation [-]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('saturations_fg.png', dpi=300)
    plt.close()


    ## SAVE EXPERIMENTAL DATA
    data = {
        'core': {
            'k': core['k'],
            'phi': core['phi']
        },
        'relPerm': infos, 
        'fluids': {
            'muw': muw,
            'mug': mug,
            'muo': muo,
            'sigma_wg': sigma_wg
        },
        'Foam_Quality_Scan_01':{
            'fg': fg.tolist(),
            'muApp': fgMuapp[:,1].tolist(),
            'velocities': {
                'ug': ug.tolist(),
                'uw': uw.tolist(),
                'uo': uo.tolist(),
                'ut': ut_calc.tolist()
            },
            'gradP': gradP.tolist(),
            'phase_saturations': {
                'Sw': Sw.tolist(),
                'Sg': Sg.tolist(),
                'So': So.tolist()
            }
        }
    }

    # Save to file
    with open('experimentalData.json', 'w') as f:
        json.dump(data, f, indent=4)

    # with open('data.json', 'r') as f:
    #     loaded_data = json.load(f)

    # # print(loaded_data)
    # core_loaded = loaded_data['core']
    