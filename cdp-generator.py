import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

def calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, e_rate):

    # Anzahl Werte
    n = 20

    # Arrays Werte nach Strain Rate
    sigma_c_imp = []
    sigma_il_imp = []
    e_il_imp = []
    sigma_t_imp = []
    sigma_t_exp_imp = []
    w_t_imp = []
    e_crt_imp = []

    # Dehnung
    e_c = np.linspace(0, e_clim*3, n)

    # Werte nach FIB2010
    f_ck = f_cm - 8
    f_ctm = 0.3 * (f_ck)**(2/3)
    if f_ck > 50:
        f_ctm = 2.12 * np.log(1+0.1*(f_cm))

    # E-Modul, Annahme Quartzite aggregates, anpassen wenn nötig
    alpha_E = 1.0
    E_ci = 21500*alpha_E*(f_cm/10)**(1/3)
    alpha = (0.8 + 0.2*f_cm/88)
    if alpha > 1:
        alpha = 1
    E_c = alpha*E_ci

    # Zug
    G_f = 73 * f_cm**0.18 / 1000  # N/mm

    # CDP Eigenschaften
    v_c0 = 0.5  # Poissons at Peak Engineering Stress
    v_ce = 8e-6*f_cm**2 + 0.0002*f_cm + 0.138  # Poisson Elastic
    winkel = np.arctan(
        6*(v_c0-v_ce)/(3*E_c*e_c1/f_cm+2*(v_c0-v_ce)-3))*180/np.pi
    fbfc = 1.57*f_cm**(-0.09)
    K_c = 0.71*f_cm**(-0.025)

    # Schubmodul
    G_c = E_c / (2*(1+v_ce))

    # Element Size
    l_0 = 0.4*E_c*1e6*G_f*1000/(f_ctm*1e6)**2  # m

    for l in range(len(e_rate)):

        f_cm_imp = f_cm*(e_rate[l]/0.00003)**0.014
        if e_rate[l] > 30:
            f_cm_imp = 0.012*f_cm*(e_rate[l]/0.00003)**(1/3)
        elif e_rate[l] == 0:
            f_cm_imp = f_cm

        f_ctm_imp = f_ctm*(e_rate[l]/1e-6)**0.018
        if e_rate[l] > 10:
            f_ctm_imp = 0.0062*f_ctm*(e_rate[l]/1e-6)**(1/3)
        elif e_rate[l] == 0:
            f_ctm_imp = f_ctm

        f_cel_imp = f_cm_imp * 0.4

        e_c1_imp = e_c1*(e_rate[l]/0.00003)**0.02
        if e_rate[l] == 0:
            e_c1_imp = e_c1

        E_ci_imp = E_ci*(e_rate[l]/0.00003)**0.025
        if e_rate[l] == 0:
            E_ci_imp = E_ci

        E_c1_imp = f_cm_imp/e_c1_imp

        # Druck nach CEB-90
        eta_E = E_ci_imp / E_c1_imp
        e_clim = e_c1_imp*(0.5*(0.5*eta_E+1) +
                           (0.25*((0.5*eta_E+1)**2)-0.5)**0.5)
        eta = e_c / e_c1_imp
        eta_lim = e_clim / e_c1_imp
        sigma_c = np.array(eta)
        xi = 4*(eta_lim**2*(eta_E-2)+2*eta_lim-eta_E) / \
            ((eta_lim*(eta_E-2)+1)**2)
        for i in range(len(e_c)):
            sigma_c[i] = (eta_E*e_c[i]/e_c1_imp-(e_c[i]/e_c1_imp)
                          ** 2)/(1+(eta_E-2)*(e_c[i]/e_c1_imp))*f_cm_imp
            if e_c[i] > e_clim:
                sigma_c[i] = f_cm_imp/((xi/eta_lim-2/(eta_lim**2)) *
                                       ((e_c[i]/e_c1_imp)**2)+(4/eta_lim-xi)*e_c[i]/e_c1_imp)
        sigma_c_imp.append(sigma_c)

        # Inelastic Strain
        e_il = []
        sigma_il = []
        aux = False
        for i in range(len(e_c)):
            if sigma_c[i] > f_cel_imp:
                if e_c[i] - sigma_c[i] / E_c > 0:
                    if not aux:
                        aux = True
                        e_il.append(0)
                    else:
                        e_il.append(e_c[i] - sigma_c[i] / E_c)
                    sigma_il.append(sigma_c[i])
            elif e_c[i] > e_c1_imp:
                e_il.append(e_c[i] - sigma_c[i] / E_c)
                sigma_il.append(sigma_c[i])
        sigma_il_imp.append(np.array(sigma_il))
        e_il_imp.append(np.array(e_il))

        # Zug
        w_rate = e_rate[l]*l_ch
        G_f_imp = G_f * (w_rate/0.01)**0.08  # N/mm #Li et al.
        print(G_f_imp)
        if w_rate > 200:
            b_g = (200/0.01)**(0.08-0.62)
            G_f_imp = G_f * b_g * (w_rate/0.01)**0.62
        elif e_rate[l] == 0:
            G_f_imp = G_f
        # print (G_f_imp)

        # Bilinear nach FIB2010
        # w_1 = G_f_imp / f_ctm_imp
        w_1 = (G_f / f_ctm) * (e_rate[l] / 1e-6)**.02
        if e_rate[l] == 0:
            w_1 = (G_f / f_ctm)
        # w_c = 5*G_f_imp / f_ctm_imp
        w_c = (5*G_f / f_ctm) * (e_rate[l] / 1e-6)**.02
        if e_rate[l] == 0:
            w_c = (5*G_f / f_ctm)
        w_t = np.linspace(0, w_c, n)
        sigma_t = np.linspace(0, 1, n)
        for i in range(len(w_t)):
            if w_t[i] < w_1:
                sigma_t[i] = f_ctm_imp*(1 - 0.8 * w_t[i]/w_1)
            else:
                sigma_t[i] = f_ctm_imp*(0.25 - 0.05 * w_t[i]/w_1)
        w_t_imp.append(w_t)
        sigma_t_imp.append(sigma_t)

        # Generalized Power Law
        G_f_imp = G_f
        if e_rate[l] != 0:
            G_f_imp = -1*np.trapz(w_t, sigma_t)
        n_exp = G_f_imp/(f_ctm_imp*w_c-G_f_imp)
        sigma_t_exp = f_ctm_imp * (1 - (w_t/w_c)**n_exp)
        sigma_t_exp_imp.append(sigma_t_exp)

        # Dehnung
        e_crt = w_t / l_ch
        e_crt_imp.append(e_crt)

    # Interp
    for i in range(len(sigma_il_imp)):
        if i != 0:
            sigma_il_imp[i] = np.interp(
                e_il_imp[0], e_il_imp[i], sigma_il_imp[i])
            e_il_imp[i] = e_il_imp[0]
    for i in range(len(sigma_t_imp)):
        if i != 0:
            sigma_t_imp[i] = np.interp(w_t_imp[0], w_t_imp[i], sigma_t_imp[i])
            sigma_t_exp_imp[i] = np.interp(
                w_t_imp[0], w_t_imp[i], sigma_t_exp_imp[i])
            e_crt_imp[i] = e_crt_imp[0]
            w_t_imp[i] = w_t_imp[0]

    # Inelastic Damage
    damage_c = []
    sigma_il = sigma_il_imp[0]
    e_il = e_il_imp[0]
    for i in range(len(e_il)):
        damage = 0
        if sigma_il[i] < f_cm:
            damage = 1-sigma_il[i]/f_cm
        damage_c.append(damage)
    damage_c = np.array(damage_c)
    damage_c[np.gradient(damage_c) < 0] = 0

    # Inelastic Damage Zug
    # Bilinear
    damage_t = []
    sigma_t = sigma_t_imp[0]
    e_crt = e_crt_imp[0]
    for i in range(len(e_crt)):
        damage = 1-sigma_t[i]/f_ctm
        # damage = 0.2*e_crt[i]*E_c/(0.2*e_crt[i]*E_c+sigma_t[i])
        damage_t.append(damage)
    damage_t = np.array(damage_t)
    # Generalized Power Law
    damage_t_exp = []
    sigma_t_exp = sigma_t_exp_imp[0]
    for i in range(len(e_crt)):
        damage = 1-sigma_t_exp[i]/f_ctm
        # damage = 0.2*e_crt[i]*E_c/(0.2*e_crt[i]*E_c+sigma_t_exp[i])
        damage_t_exp.append(damage)
    damage_t_exp = np.array(damage_t_exp)

    return {
        'properties': {'elasticity': E_c, 'shear': G_c, 'fracture energy': G_f, 'tensile strength': f_ctm, 'dilation angle': winkel, 'poisson': v_ce, 'Kc': K_c, 'fbfc': fbfc, 'l0': l_0},
        'compression': {'strain': e_c, 'stress': sigma_c_imp, 'inelastic strain': e_il_imp, 'inelastic stress': sigma_il_imp, 'damage': damage_c},
        'tension': {'crack opening': w_t_imp, 'stress': sigma_t_imp, 'stress exponential': sigma_t_exp_imp, 'cracking strain': e_crt_imp, 'cracking stress': sigma_t_imp, 'damage': damage_t, 'damage exponential': damage_t_exp}
    }


def calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch):

    # Anzahl Werte
    n = 40

    # Arrays Werte nach Temperatur
    sigma_c_imp = []
    sigma_il_imp = []
    e_il_imp = []
    sigma_t_imp = []
    sigma_t_exp_imp = []
    w_t_imp = []
    e_crt_imp = []
    e_c_imp = []

    # Eurocode
    temp_table = np.array([
        [20,   1.00, 0.0025, 0.0200],
        [100,  1.00, 0.0040, 0.0225],
        [200,  0.95, 0.0055, 0.0250],
        [300,  0.85, 0.0070, 0.0275],
        [400,  0.75, 0.0100, 0.0300],
        [500,  0.60, 0.0150, 0.0325],
        [600,  0.45, 0.0250, 0.0350],
        [700,  0.30, 0.0250, 0.0375],
        [800,  0.15, 0.0250, 0.0400],
        [900,  0.08, 0.0250, 0.0425],
        [1000, 0.04, 0.0250, 0.0450],
        [1100, 0.01, 0.0250, 0.0475],
    ])

    # Werte nach FIB2010
    f_ck = f_cm - 8
    f_ctm = 0.3 * (f_ck)**(2/3)
    if f_ck > 50:
        f_ctm = 2.12 * np.log(1+0.1*(f_cm))

    # E-Modul, Annahme Quartzite aggregates, anpassen wenn nötig
    alpha_E = 1.0
    E_ci = 21500*alpha_E*(f_cm/10)**(1/3)
    alpha = (0.8 + 0.2*f_cm/88)
    if alpha > 1:
        alpha = 1
    E_c = alpha*E_ci
    E_c1 = f_cm/e_c1

    # Zug
    G_f = 73 * f_cm**0.18 / 1000  # N/mm

    # CDP Eigenschaften
    v_c0 = 0.5  # Poissons at Peak Engineering Stress
    v_ce = 8e-6*f_cm**2 + 0.0002*f_cm + 0.138  # Poisson Elastic
    winkel = np.arctan(
        6*(v_c0-v_ce)/(3*E_c*e_c1/f_cm+2*(v_c0-v_ce)-3))*180/np.pi
    fbfc = 1.57*f_cm**(-0.09)
    K_c = 0.71*f_cm**(-0.025)

    # Column labels for reference
    T, f_ratio, eps_c, eps_cu = temp_table[:,
                                           0], temp_table[:, 1], temp_table[:, 2], temp_table[:, 3]

    # Schubmodul
    G_c = E_c / (2*(1+v_ce))

    # Element Size
    l_0 = 0.4*E_c*1e6*G_f*1000/(f_ctm*1e6)**2  # m

    for l in range(len(T)):

        e_c = np.linspace(0, eps_cu[l]*2, n)
        f_ck_imp = f_ratio[l]*f_ck
        f_cm_imp = f_ratio[l]*f_cm
        if T[l]<=100:
            f_ctm_imp_EC = f_ctm
        else:
            f_ctm_imp_EC = f_ctm*(1-1*(T[l]-100)/500)
            if f_ctm_imp_EC<0:
                f_ctm_imp_EC = 0
        f_ctm_imp = 0.3 * (f_ck_imp)**(2/3)
        print('\nTensile Strength [N/mm²], at '+str(T[l])+' [°C]: '+str(f_ctm_imp))
        print('\nTensile Strength [N/mm²] (EC2), at '+str(T[l])+' [°C]: '+str(f_ctm_imp_EC))
        if f_ck > 50:
            f_ctm_imp = 2.12 * np.log(1+0.1*(f_cm_imp))

        f_cel_imp = f_cm_imp * 0.4

        e_c1_imp = eps_c[l]

        alpha_E = 1.0
        #E_ci_imp = 21500*alpha_E*(f_cm_imp/10)**(1/3)
        #E_c1_imp = f_cm_imp/e_c1_imp
        E_c1_imp = (f_ratio[l]/eps_c[l])/(f_ratio[1]/eps_c[1])*E_c1
        k_imp = -0.0193*f_ck_imp+2.6408
        E_ci_imp = E_c1_imp*k_imp
        print('\nE-Modul sekant [N/mm²], at '+str(T[l])+' [°C]: '+str(E_c1_imp))
        print('\nE-Modul tangent [N/mm²], at '+str(T[l])+' [°C]: '+str(E_ci_imp))

        # Druck nach CEB-90
        eta_E = E_ci_imp / E_c1_imp
        e_clim = e_c1_imp*(0.5*(0.5*eta_E+1) +
                           (0.25*((0.5*eta_E+1)**2)-0.5)**0.5)
        eta = e_c / e_c1_imp
        eta_lim = e_clim / e_c1_imp
        sigma_c = np.array(eta)
        xi = 4*(eta_lim**2*(eta_E-2)+2*eta_lim-eta_E) / \
            ((eta_lim*(eta_E-2)+1)**2)
        for i in range(len(e_c)):
            sigma_c[i] = (eta_E*e_c[i]/e_c1_imp-(e_c[i]/e_c1_imp)
                          ** 2)/(1+(eta_E-2)*(e_c[i]/e_c1_imp))*f_cm_imp
            if e_c[i] > e_clim:
                sigma_c[i] = f_cm_imp/((xi/eta_lim-2/(eta_lim**2)) *
                                       ((e_c[i]/e_c1_imp)**2)+(4/eta_lim-xi)*e_c[i]/e_c1_imp)
        sigma_c_imp.append(sigma_c)
        e_c_imp.append(e_c)

        # Inelastic Strain
        e_il = []
        sigma_il = []
        aux = False
        for i in range(len(e_c)):
            if sigma_c[i] > f_cel_imp:
                if e_c[i] - sigma_c[i] / E_c1_imp > 0:
                    if not aux:
                        aux = True
                        e_il.append(0)
                    else:
                        e_il.append(e_c[i] - sigma_c[i] / E_c1_imp)
                    sigma_il.append(sigma_c[i])
            elif e_c[i] > e_c1_imp:
                e_il.append(e_c[i] - sigma_c[i] / E_c1_imp)
                sigma_il.append(sigma_c[i])
        sigma_il_imp.append(np.array(sigma_il))
        e_il_imp.append(np.array(e_il))

        # Zug
        G_f_imp = 73 * f_cm_imp**0.18 / 1000  # N/mm
        # print (G_f_imp)

        # Bilinear nach FIB2010
        w_1 = G_f_imp / f_ctm_imp
        w_c = 5*G_f_imp / f_ctm_imp
        w_t = np.linspace(0, w_c, n)
        sigma_t = np.linspace(0, 1, n)
        for i in range(len(w_t)):
            if w_t[i] < w_1:
                sigma_t[i] = f_ctm_imp*(1 - 0.8 * w_t[i]/w_1)
            else:
                sigma_t[i] = f_ctm_imp*(0.25 - 0.05 * w_t[i]/w_1)
        w_t_imp.append(w_t)
        sigma_t_imp.append(sigma_t)

        # Generalized Power Law
        n_exp = G_f_imp/(f_ctm_imp*w_c-G_f_imp)
        sigma_t_exp = f_ctm_imp * (1 - (w_t/w_c)**n_exp)
        sigma_t_exp_imp.append(sigma_t_exp)

        # Dehnung
        e_crt = w_t / l_ch
        e_crt_imp.append(e_crt)

    # Interp
    for i in range(len(sigma_il_imp)):
        if i != 0:
            sigma_il_imp[i] = np.interp(
                e_il_imp[0], e_il_imp[i], sigma_il_imp[i])
            e_il_imp[i] = e_il_imp[0]
    for i in range(len(sigma_t_imp)):
        if i != 0:
            sigma_t_imp[i] = np.interp(w_t_imp[0], w_t_imp[i], sigma_t_imp[i])
            sigma_t_exp_imp[i] = np.interp(
                w_t_imp[0], w_t_imp[i], sigma_t_exp_imp[i])
            e_crt_imp[i] = e_crt_imp[0]
            w_t_imp[i] = w_t_imp[0]

    # Inelastic Damage
    damage_c = []
    sigma_il = sigma_il_imp[0]
    e_il = e_il_imp[0]
    for i in range(len(e_il)):
        damage = 0
        if sigma_il[i] < f_cm:
            damage = 1-sigma_il[i]/f_cm
        damage_c.append(damage)
    damage_c = np.array(damage_c)
    damage_c[np.gradient(damage_c) < 0] = 0

    # Inelastic Damage Zug
    # Bilinear
    damage_t = []
    sigma_t = sigma_t_imp[0]
    e_crt = e_crt_imp[0]
    for i in range(len(e_crt)):
        damage = 1-sigma_t[i]/f_ctm
        # damage = 0.2*e_crt[i]*E_c/(0.2*e_crt[i]*E_c+sigma_t[i])
        damage_t.append(damage)
    damage_t = np.array(damage_t)
    # Generalized Power Law
    damage_t_exp = []
    sigma_t_exp = sigma_t_exp_imp[0]
    for i in range(len(e_crt)):
        damage = 1-sigma_t_exp[i]/f_ctm
        # damage = 0.2*e_crt[i]*E_c/(0.2*e_crt[i]*E_c+sigma_t_exp[i])
        damage_t_exp.append(damage)
    damage_t_exp = np.array(damage_t_exp)

    return {
        'properties': {'elasticity': E_c, 'shear': G_c, 'fracture energy': G_f, 'tensile strength': f_ctm, 'dilation angle': winkel, 'poisson': v_ce, 'Kc': K_c, 'fbfc': fbfc, 'l0': l_0},
        'compression': {'strain': e_c, 'stress': sigma_c_imp, 'inelastic strain': e_il_imp, 'inelastic stress': sigma_il_imp, 'damage': damage_c, 'strain temp': e_c_imp},
        'tension': {'crack opening': w_t_imp, 'stress': sigma_t_imp, 'stress exponential': sigma_t_exp_imp, 'cracking strain': e_crt_imp, 'cracking stress': sigma_t_imp, 'damage': damage_t, 'damage exponential': damage_t_exp}
    }

def plot_curve(x, y, title, xlabel, ylabel, var, style=None):
    if style is None:
        style = {"color": "#1f77b4", "marker": "o"}
    
    # strain rate
    if var < 1:
        # plt.plot(x, y, marker='o', label = r'$\dot{\varepsilon}=$'+str(var)+ ' [s$^{-1}$]')
        plt.plot(x, y, label=r'$\dot{\varepsilon}=$' +
                 str(var) + ' [s$^{-1}$]', **style, linestyle='-')
    # temperature
    else:
        # plt.plot(x, y, marker='o', label = r'$T=$'+str(var)+ ' [°C]')
        plt.plot(x, y, label=r'$T=$'+str(var) +
                 ' [°C]', **style, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def for_plot(x, y, title, xlabel, ylabel, var):
    custom_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#a55194", "#393b79"]
    custom_markers = ['o', 's', '^', 'v', 'D',
                      'P', '*', 'X', 'h', '<', '>', '8']
    custom_style = cycle([
        {"color": c, "marker": m}
        for c, m in zip(custom_colors, custom_markers)])
    plt.figure()
    if len(x) != len(y):
        for i in range(len(y)):
            style = next(custom_style)
            plot_curve(x, y[i], title, xlabel, ylabel, var[i], style)
    else:
        for i in range(len(y)):
            style = next(custom_style)
            plot_curve(x[i], y[i], title, xlabel, ylabel, var[i], style)
    plt.grid()
    plt.show()
    plt.legend(loc="upper right")


if __name__ == "__main__":

    # Inputs with default values
   f_cm = input(
       "Enter the compressive strength of the concrete (MPa) [Default: 28]: ")
   e_c1 = input(
       "Enter the strain at maximum compressive strength c_i [Default: 0.0022]: ")
   e_clim = input("Enter the strain at ultimate state [Default: 0.0035]: ")
   l_ch = input(
       "Enter the characteristic element length of the mesh (mm) [Default: 1]: ")
   e_rate = input(
       "Enter the strain rates additional to 0/s, separated by a coma [Default: 2,30,100]: ")
   while True:
       temp_input = input(
           "Temperature Dependent Data? (y/n) [Default: n]: ").strip().lower()
       if temp_input in ("y", "n", ""):
           temp = 0 if temp_input == "y" else 1
           break
       else:
           print("Invalid input. Please enter 'y' or 'n'.")

   # Assign default values if no input is provided
   f_cm = float(f_cm.strip()) if f_cm.strip() else 28
   e_c1 = float(e_c1.strip()) if e_c1.strip() else 0.0022
   e_clim = float(e_clim.strip()) if e_clim.strip() else 0.0035
   l_ch = float(l_ch.strip()) if l_ch.strip() else 1
   e_rate = [0]+list(map(float, e_rate.strip().split(','))
                     ) if e_rate.strip() else [0, 2, 30, 100]
   temps = np.array([20, 100, 200, 300, 400, 500,
                    600, 700, 800, 900, 1000, 1100])

   # Calculate stress-strain relationships
   if temp:
       results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, e_rate)
   else:
       results = calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch)

   # Plot
   if temp:
       var = e_rate
       for_plot(results['compression']['strain'], results['compression']['stress'],
                'Compressive Strain - Compressive Stress', 'Compressive Strain [-]', 'Compressive Stress [MPa]', var)
       for_plot(results['compression']['inelastic strain'], results['compression']['inelastic stress'],
                'Compressive Inelastic Strain - Compressive Stress', 'Compressive Inelastic Strain [-]', 'Compressive Stress [MPa]', var)
   else:
       var = temps
       for_plot(results['compression']['strain temp'], results['compression']['stress'],
                'Compressive Strain - Compressive Stress', 'Compressive Strain [-]', 'Compressive Stress [MPa]', var)
       for_plot(results['compression']['inelastic strain'], results['compression']['inelastic stress'],
                'Compressive Inelastic Strain - Compressive Stress', 'Compressive Inelastic Strain [-]', 'Compressive Stress [MPa]', var)

   plt.figure()
   plot_curve(results['compression']['inelastic strain'][0], results['compression']['damage'],
              'Compressive Damage', 'Compressive Inelastic Strain [-]', 'Damage [-]', var[0])
   plt.grid()
   plt.show()
   for_plot(results['tension']['crack opening'], results['tension']['stress'],
            'Crack Opening - Tensile Stress (Bilinear)', 'Crack Opening [mm]', 'Cracking Stress [MPa]', var)
   for_plot(results['tension']['crack opening'], results['tension']['stress exponential'],
            'Crack Opening - Tensile Stress (Power Law)', 'Crack Opening [mm]', 'Cracking Stress [MPa]', var)
   for_plot(results['tension']['cracking strain'], results['tension']['stress'],
            'Cracking Strain - Tensile Stress', 'Cracking Strain [-]', 'Cracking Stress [MPa]', var)
   for_plot(results['tension']['cracking strain'], results['tension']['stress exponential'],
            'Cracking Strain - Tensile Stress (Power Law)', 'Cracking Strain [-]', 'Cracking Stress [MPa]', var)
   plt.figure()
   plot_curve(results['tension']['cracking strain'][0], results['tension']['damage'],
              'Tension Damage (Bilinear)', 'Cracking Strain [-]', 'Damage [-]', var[0])
   plt.grid()
   plt.show()
   plt.figure()
   plot_curve(results['tension']['cracking strain'][0], results['tension']['damage exponential'],
              'Tension Damage (Power Law)', 'Cracking Strain [-]', 'Damage [-]', var[0])
   plt.grid()
   plt.show()
   print('\nCompressive Strength: '+str(f_cm)+' [MPa]')
   print('Tensile Strength: ' +
         str(round(results['properties']['tensile strength'], 2))+' [MPa]')
   print('Elasticity Modulus: ' +
         str(round(results['properties']['elasticity'], 2))+' [MPa]')
   print('Poisson: '+str(round(results['properties']['poisson'], 2))+' [-]')
   print('Shear Modulus: ' +
         str(round(results['properties']['shear'], 2))+' [MPa]')
   print('Fracture Energy: ' +
         str(round(results['properties']['fracture energy'], 2))+' [N/mm]')
   print('CDP Dilation angle: ' +
         str(round(results['properties']['dilation angle'], 2))+' [°]')
   print('CDP fb/fc: '+str(round(results['properties']['fbfc'], 2))+' [-]')
   print('CDP Kc: '+str(round(results['properties']['Kc'], 2))+' [-]')
   print('Max. Mesh Size: '+str(round(results['properties']['l0'], 2))+' [m]')

   # Export

   e_rate_export = np.array([])
   e_rate_export_el = np.array([])
   compressive_stress_el = np.array([])
   compressive_stress = np.array([])
   compressive_strain_el = np.array([])
   compressive_strain = np.array([])
   e_rate_export_tension = np.array([])
   tension_stress = np.array([])
   tension_strain = np.array([])
   tension_crack = np.array([])
   tension_stress_exp = np.array([])
   for i in range(len(var)):
       e_rate_export_el = np.concatenate(
           (e_rate_export_el, np.ones(len(results['compression']['strain']))*var[i]))
       e_rate_export = np.concatenate((e_rate_export, np.ones(
           len(results['compression']['inelastic strain'][0]))*var[i]))
       compressive_stress_el = np.concatenate(
           (compressive_stress_el, results['compression']['stress'][i]))
       compressive_stress = np.concatenate(
           (compressive_stress, results['compression']['inelastic stress'][i]))
       compressive_strain_el = np.concatenate(
           (compressive_strain_el, results['compression']['strain']))
       compressive_strain = np.concatenate(
           (compressive_strain, results['compression']['inelastic strain'][i]))
       e_rate_export_tension = np.concatenate((e_rate_export_tension, np.ones(
           len(results['tension']['cracking strain'][0]))*var[i]))
       tension_stress = np.concatenate(
           (tension_stress, results['tension']['stress'][i]))
       tension_strain = np.concatenate(
           (tension_strain, results['tension']['cracking strain'][i]))
       tension_crack = np.concatenate(
           (tension_crack, results['tension']['crack opening'][i]))
       tension_stress_exp = np.concatenate(
           (tension_stress_exp, results['tension']['stress exponential'][i]))

   compression_strain_el = pd.DataFrame({
       'Compressive Stress [MPa]': compressive_stress_el,
       'Strain [-]': compressive_strain_el,
       'Strain Rate [1/s]': e_rate_export_el
   })
   compression_strain = pd.DataFrame({
       'Compressive Stress [MPa]': compressive_stress,
       'Inelastic Strain [-]': compressive_strain,
       'Strain Rate [1/s]': e_rate_export
   })
   compression_damage = pd.DataFrame({
       'Damage [-]': results['compression']['damage'],
       'Inelastic Strain [-]': results['compression']['inelastic strain'][0]
   })
   tension_cracking = pd.DataFrame({
       'Tension Stress [MPa]': tension_stress,
       'Crack Opening [mm]': tension_crack,
       'Strain Rate [1/s]': e_rate_export_tension
   })
   tension_cracking_strain = pd.DataFrame({
       'Tension Stress [MPa]': tension_stress,
       'Cracking Strain [-]': tension_strain,
       'Strain Rate [1/s]': e_rate_export_tension
   })
   tension_damage = pd.DataFrame({
       'Damage [-]': results['tension']['damage'],
       'Cracking Strain [-]': results['tension']['cracking strain'][0]
   })
   tension_cracking_power = pd.DataFrame({
       'Tension Stress [MPa]': tension_stress_exp,
       'Cracking Strain [-]': tension_strain,
       'Strain Rate [1/s]': e_rate_export_tension
   })
   tension_cracking_strain_power = pd.DataFrame({
       'Tension Stress [MPa]': tension_stress_exp,
       'Cracking Strain [-]': tension_strain,
       'Strain Rate [1/s]': e_rate_export_tension
   })
   tension_damage_power = pd.DataFrame({
       'Damage [-]': results['tension']['damage exponential'],
       'Cracking Strain [-]': results['tension']['cracking strain'][0]
   })
   excel_file_path = 'CDP-Results.xlsx'
   with pd.ExcelWriter(excel_file_path) as writer:
       compression_strain_el.to_excel(
           writer, sheet_name="Compression Stress-Strain", index=False)
       compression_strain.to_excel(
           writer, sheet_name="Compression Inl.Strain", index=False)
       compression_damage.to_excel(
           writer, sheet_name="Compression Damage", index=False)
       tension_cracking.to_excel(
           writer, sheet_name="Tension Cracking", index=False)
       tension_cracking_strain.to_excel(
           writer, sheet_name="Tension Cr.Strain", index=False)
       tension_damage.to_excel(
           writer, sheet_name="Tension Damage", index=False)
       tension_cracking_power.to_excel(
           writer, sheet_name="Tension Cracking Power", index=False)
       tension_cracking_strain_power.to_excel(
           writer, sheet_name="Tension Cr.Strain Power", index=False)
       tension_damage_power.to_excel(
           writer, sheet_name="Tension Damage Power", index=False)
