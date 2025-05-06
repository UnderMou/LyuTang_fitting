import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.optimize import minimize_scalar

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
    scatter = plt.scatter(uw*283465, ug*283465, c=np.abs(gradP)*4.419e-5, cmap='viridis', edgecolor='k')
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'$\nabla p$ [psi/ft]')
    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('uwug_gradP.png', dpi=300)
    plt.close()

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
    