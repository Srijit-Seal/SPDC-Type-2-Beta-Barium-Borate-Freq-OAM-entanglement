import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy import ndimage

# #############################################################################################GAUSSIAN PART OF THE CODE-----------------------------------------------------------------------
def gaussian(N, win, w0):
    X = np.linspace(-win / 2, +win / 2, N)
    Y = np.linspace(-win / 2, +win / 2, N)
    X, Y = np.meshgrid(X, Y)
    Gauss = np.exp((-(X ** 2 + Y ** 2)) / (w0 ** 2))

    plt.figure(figsize=(7, 7))
    plt.imshow(Gauss, extent=[X.min() * (1000), X.max() * (1000), Y.min() * (1000), Y.max() * (1000)], cmap='turbo', origin='lower')
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Input Gaussian Pump Profile")
    plt.colorbar(label="Normalized amplitude")

    return Gauss, X, Y


####################################################################################################BBO PART OF THE CODE--------------------------------------------------------------------
c = 3e8
pi = np.pi


def n_o(lam_um):
    return np.sqrt(2.7359 + 0.01878 / (lam_um ** 2 - 0.01822) - 0.01354 * lam_um ** 2)


def n_e(lam_um):
    return np.sqrt(2.3753 + 0.01224 / (lam_um ** 2 - 0.01667) - 0.01516 * lam_um ** 2)


def n_e_eff(lam_um, theta_deg):
    theta = np.radians(theta_deg)
    return 1.0 / np.sqrt((np.cos(theta) / n_o(lam_um)) ** 2 + (np.sin(theta) / n_e(lam_um)) ** 2)


def bbo(lam_pump_nm, theta, l):
    lam_pump_um = lam_pump_nm / 1000.0
    lam_signal_um = 2.0 * lam_pump_um

    w_p = 2 * pi * c / (lam_pump_um * 1e-6)
    w_s = 2 * pi * c / (lam_signal_um * 1e-6)
    w_i = 2 * pi * c / (lam_signal_um * 1e-6)
    no_s = n_o(lam_signal_um)
    ne_i = n_e_eff(lam_signal_um, theta)
    ne_p = n_e_eff(lam_pump_um, theta)
    print("no(810) =", no_s)
    print("ne(810) =", n_e(lam_signal_um))
    print("neff pump =", ne_p)
    print("theta =", theta)
    print("omega p",w_p)
    return {'omega_s': w_s,'omega_i': w_i,'omega_p': w_p,'no_signal': no_s,'ne_idler': ne_i,'ne_pump': ne_p,'l_crystal': l,'cut_angle_deg': theta,}

def walkoff_angle(lam_um, cut_angle_deg):
    theta = np.radians(cut_angle_deg)
    no = n_o(lam_um)
    ne = n_e(lam_um)
    neff = 1.0 / np.sqrt((np.cos(theta) / no) ** 2 + (np.sin(theta) / ne) ** 2)
    tan_rho = 0.5 * neff**2 * np.sin(2*theta) * (1/ne**2 - 1/no**2)
    return np.arctan(tan_rho)

def extI(psi, lam_um):
    no = n_o(lam_um)
    ne = n_e(lam_um)
    neff = 1.0 / np.sqrt((np.cos(psi)**2)/(no**2) + (np.sin(psi)**2)/(ne**2))
    return neff
########################################################################################################################
def angle_to_optic_axis(KX, KY, kz, oa_x, oa_y, oa_z):

    ux = KX[None,:,:]
    uy = KY[None,:,:]
    uz = kz

    norm = np.sqrt(ux**2 + uy**2 + uz**2)

    ux = ux/norm
    uy = uy/norm
    uz = uz/norm

    cospsi = ux*oa_x + uy*oa_y + uz*oa_z

    cospsi = np.clip(cospsi, -1, 1)
    print(np.min(cospsi))
    print(np.max(cospsi))
    psi = np.arccos(cospsi)
    return psi
#######################################################################################################################
def phase_match2(KX,KY,omg,bbo_out):
    theta_cut=np.radians(bbo_out['cut_angle_deg'])
    #optical axis description assumed along xz plane
    oaX=np.sin(theta_cut)
    oaY=0.0
    oaZ=np.cos(theta_cut)
    ############frequencies################
    w_s=omg[:,None,None]
    #w_i = omg[:, None, None]
    w_p=bbo_out['omega_p']
    w_i=w_p-w_s
    #########wavelengths###################
    lam_i_um = (2 * np.pi * c / w_i) * 1e6
    lam_s_um = (2 * np.pi * c / w_s) * 1e6
    lam_p_um = (2 * np.pi * c / w_p) * 1e6
    mid = Nw // 2


    no_s=n_o(lam_s_um)

    k_s = w_s * no_s / c
    print("k_s shape =", k_s.shape)

    k_i0 = w_i * bbo_out['ne_idler'] / c
    k_p0 = w_p * bbo_out['ne_pump'] / c

    kz_i0 = np.sqrt(np.maximum(k_i0 ** 2 - KX[None, :, :] ** 2 - KY[None, :, :] ** 2, 0))
    kz_p0 = np.sqrt(np.maximum(k_p0 ** 2 - KX[None, :, :] ** 2 - KY[None, :, :] ** 2, 0))
    print("KX center =", KX[N // 2, N // 2])
    print("KY center =", KY[N // 2, N // 2])
    idx0 = np.unravel_index(np.argmin(KX ** 2 + KY ** 2), KX.shape)


    psi_i = angle_to_optic_axis(KX, KY, kz_i0, oaX, oaY, oaZ)
    psi_p = theta_cut
    neff_i = extI( psi_i, lam_i_um)

    print("psi center (deg) =", np.degrees(psi_i[mid, N // 2, N // 2]))
    print("neff_i center =",neff_i[mid, N // 2, N // 2])
    neff_p = n_e_eff(lam_p_um, bbo_out['cut_angle_deg'])
    k_i = w_i * neff_i / c
    k_p = w_p * neff_p / c
    kz_s = np.sqrt(np.maximum(k_s ** 2 - KX[None, :, :] ** 2 - KY[None, :, :] ** 2, 0))
    kz_i = np.sqrt(np.maximum(k_i ** 2 - KX[None, :, :] ** 2 - KY[None, :, :] ** 2, 0))
    kz_p = k_p*np.ones_like(kz_i)
    del_k = kz_p - kz_s - kz_i
    phi = np.sinc(del_k * bbo_out['l_crystal'] / (2 * np.pi))
    return del_k, k_i, k_p, kz_i, kz_p
#######################################################################################################Split Step Method-------------------------------------------------------------------
def linear(Ea, Ha):

    E_freq = np.fft.fftshift(np.fft.fft2(Ea, axes=(-2, -1)),axes=(-2, -1))
    E_freq = E_freq * Ha
    E_freq = np.fft.ifftshift(E_freq,axes=(-2, -1))
    return np.fft.ifft2(E_freq, axes=(-2, -1))

def non_linear_step(kappa_x, del_k, z, conj_other_vac, dl):
    gen_term = kappa_x * conj_other_vac                      # real-space generation term
    gen_k = np.fft.fftshift(np.fft.fft2(gen_term, axes=(-2,-1)), axes=(-2,-1))
    gen_k *= np.exp(-1j * del_k * z)                          # del_k correctly applied in k-space
    gen_real = np.fft.ifft2(np.fft.ifftshift(gen_k, axes=(-2,-1)), axes=(-2,-1))
    return gen_real * dl
########################################################################################################################
def ssf(Gs, Es_vac, Ei_vac, Es_out, Ei_out, bbo_out, steps, X, omg,cut_theta):
    Nx, Ny = len(Gs), len(Gs[0])
    # wave number calculation
    c = 3e8

    k_s = omg * bbo_out['no_signal'] / c
    k_i = omg * bbo_out['ne_idler'] / c
    k_s = (omg * bbo_out['no_signal'] / c)[:, None, None]
    k_i = ((bbo_out['omega_p'] - omg) * bbo_out['ne_idler'] / c)[:, None, None]
    #del_k =( k_p - k_s - k_i)

    # --------------------------------
    w_p = bbo_out.get('omega_p')

    w_s = omg[:, None, None]
    w_i = w_p-w_s
    valid = (w_i > 1e12)

    w_i = np.where(valid,w_i,np.nan)
    # ---------------------------------
    chi_2 = 2e-12
    l = bbo_out.get('l_crystal')
    dl = l / steps
    # ----------------for non linear part------------------

    E0 = 4e8
    kappa_i = ((w_i ** 2) / (c ** 2 * k_i)) * chi_2 * E0 * Gs
    kappa_s = ((w_s ** 2) / (c ** 2 * k_s)) * chi_2 * E0 * Gs

    z = 0
    # --------------linear propagators---------------------

    _, Nx, Ny = Gs.shape
    dx = X[0, 1] - X[0, 0]  # actual grid spacing
    dy = dx
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Nx, dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Ny, dy))
    KX, KY = np.meshgrid(kx*23, ky*23)
    #del_k, k_i_eff, k_p_eff, kz_i, kz_p = phase_match2(KX, KY, omg, bbo_out)

    del_k, k_i_eff, k_p_eff, kz_i, kz_p = phase_match2(KX, KY, omg, bbo_out)
    
    mid = Nw // 2

    iy, ix = np.unravel_index(np.argmin(np.abs(del_k[mid])),del_k[mid].shape)

    K_radius = np.sqrt(KX[iy, ix] ** 2 +KY[iy, ix] ** 2)

    k_s_center = omg[mid] * bbo_out['no_signal'] / (3e8)

    theta_emit_deg = np.degrees(np.arcsin(K_radius / k_s_center))

    print("best pixel =", (iy, ix))
    print("K_radius =", K_radius, "m^-1")
    print("k_s =", k_s_center, "m^-1")
    print("Emission angle =", theta_emit_deg, "deg")

    Hi = np.exp(-1j * (KX ** 2 + KY ** 2) * dl / (2 * (k_i_eff+1e-20)))
    Hs = np.exp(-1j * (KX ** 2 + KY ** 2) * dl / (2 * (k_s+1e-20)))

    Hi = np.exp(-1j * (KX ** 2 + KY ** 2) * dl / (2 * (k_i_eff+1e-20)))
    Hs = np.exp(-1j * (KX ** 2 + KY ** 2) * dl / (2 * (k_s+1e-20)))
    # Hp = np.exp(-1j * (KX ** 2 + KY ** 2) * dl / (2 * k_p))

    lam_signal_um = (2 * np.pi * c / bbo_out['omega_s']) * 1e6
    rho_i = walkoff_angle(lam_signal_um, cut_theta)   # idler -- e-ray -- genuinely walks off
    rho_s = 0.0                                    # signal -- o-ray -- never walks off

    Hi = Hi * np.exp(1j * KX * rho_i * dl)
    Hs = Hs * np.exp(1j * KX * rho_s * dl)


    # -------------------------------------------------------
    # --------------checks----------------------------


    print(f"dl = {dl:.6f} m")
    print(f"kappa * dl = {np.max(np.abs(kappa_s)) * dl:.4e}")
    print(f"no_signal = {bbo_out['no_signal']:.5f}")
    print(f"ne_idler  = {bbo_out['ne_idler']:.5f}")
    print(f"ne_pump   = {bbo_out['ne_pump']:.5f}")

    print(f"any nan in kappa_s: {np.any(np.isnan(kappa_s))}")
    print(f"any nan in Hi:      {np.any(np.isnan(Hi))}")

    # ------------------------------------------------

    for step in range(steps):
        z = step * dl

        Ei_out_old = Ei_out.copy()
        Es_vac_old = Es_vac.copy()
        Es_out_old = Es_out.copy()
        Ei_vac_old = Ei_vac.copy()

        # full linear step first
        Ei_out = linear(Ei_out, Hi)
        #Es_vac = linear(Es_vac, Hs)
        Es_out = linear(Es_out, Hs)
        #Ei_vac = linear(Ei_vac, Hi)

        # full nonlinear step
        """
        Ei_out += -1j * non_linear(kappa_i, del_k, z) * np.conj(Es_vac_old) * dl
        Es_vac += -1j * non_linear(kappa_s, del_k, z) * np.conj(Ei_out_old) * dl
        Es_out += -1j * non_linear(kappa_s, del_k, z) * np.conj(Ei_vac_old) * dl
        Ei_vac += -1j * non_linear(kappa_i, del_k, z) * np.conj(Es_out_old) * dl"""
        Ei_out += -1j * non_linear_step(kappa_i, del_k, z, np.conj(Es_vac_old), dl)
        Es_out += -1j * non_linear_step(kappa_s, del_k, z, np.conj(Ei_vac_old), dl)
    I_raw_s = np.abs(Es_out) ** 2
    I_raw_i = np.abs(Ei_out) ** 2
    mid = len(omg) // 2

    # Final fields in k-space
    Es_k = np.fft.fftshift(np.fft.fft2(Es_out[mid]))

    Ei_k = np.fft.fftshift(np.fft.fft2(Ei_out[mid]))

    #Phase-matching function
    phi = np.sinc(del_k[mid] * bbo_out['l_crystal'] / (2*np.pi))

    plt.figure(figsize=(15,5))


    plt.imshow(np.log10(np.abs(Es_k) ** 2 + 1e-20),cmap='turbo',origin='lower')

    plt.contour(np.abs(phi)**2,levels=[0.7*np.max(np.abs(phi))],colors='white')

    plt.tight_layout()

    return Ei_out, Es_out, phi,del_k

##################################################################################################Post Processing Optics-----------------------------------------------------------------------------
def qwp(theta, M):
    theta = np.radians(theta)

    J11 = (np.cos(theta) ** 2) + 1j * (np.sin(theta) ** 2)
    J12 = (1 - 1j) * np.sin(theta) * np.cos(theta)

    J21 = (1 - 1j) * np.sin(theta) * np.cos(theta)
    J22 = (np.sin(theta) ** 2) + 1j * (np.cos(theta) ** 2)

    out = np.zeros_like(M, dtype=complex)

    out[0] = J11 * M[0] + J12 * M[1]
    out[1] = J21 * M[0] + J22 * M[1]

    return out

#######################################################################################################################
def polarizer(choice, M):
    if choice == 1:

        P11 = 1
        P12 = -1j
        P21 = 1j
        P22 = 1

    else:

        P11 = 1
        P12 = 1j
        P21 = -1j
        P22 = 1

    out = np.zeros_like(M, dtype=complex)

    out[0] = 0.5 * (P11 * M[0] + P12 * M[1])
    out[1] = 0.5 * (P21 * M[0] + P22 * M[1])

    return out

######################################################################################################################
# q=0.5
def q_plate(M, q, omega, Omega, X, Y):
    phi = np.arctan2(Y, X)
    l = 2 * q

    vortex_plus = np.exp(1j * l * phi[None, :, :])
    vortex_minus = np.exp(-1j * l * phi[None, :, :])

    Nw = len(omega)
    dw_bin = (omega[-1] - omega[0]) / (Nw - 1)
    shift = l * Omega
    n_bins = shift / dw_bin

    n_shift = int(round(n_bins))

    out = np.zeros_like(M, dtype=complex)
    out[0] = np.roll(vortex_minus * M[1], -n_shift, axis=0)   # -l*Omega shift
    out[1] = np.roll(vortex_plus  * M[0],  n_shift, axis=0)   # +l*Omega shift
    return out
######################################################################################################################
def linear_to_spin(M):
    out = np.zeros_like(M, dtype=complex)

    # Right circular
    out[0] = (M[0] - 1j * M[1]) / np.sqrt(2)

    # Left circular
    out[1] = (M[0] + 1j * M[1]) / np.sqrt(2)

    return out


def spin_to_linear(M):
    out = np.zeros_like(M, dtype=complex)

    out[0] = (M[0] + M[1]) / np.sqrt(2)

    out[1] = 1j * (M[0] - M[1]) / np.sqrt(2)

    return out
########################################################################################################################


def find_intersection_frequencies(Es, Ei, omg, omega_p):
    
    Nw = len(omg)
    overlap = np.zeros(Nw)

    omg_i_target = omega_p - omg

    for idx_s, w_s in enumerate(omg):
        # Find the idler frequency closest to omega_p - w_s
        idx_i = np.argmin(np.abs(omg - omg_i_target[idx_s]))
        # Spatial overlap integral of intensities
        Es_field = Es[idx_s]          # (N, N)
        Ei_field = Ei[idx_i]          # (N, N)
        overlap[idx_s] = np.sum(np.abs(Es_field)**2 * np.abs(Ei_field)**2)

    # Find peaks (we expect two symmetric peaks)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(overlap, height=0.1 * overlap.max())
    if len(peaks) < 2:
        # Fallback: take the two largest values
        peak_indices = np.argsort(overlap)[-2:][::-1]
    else:
        # Sort peaks by height and take the top two
        peak_indices = peaks[np.argsort(overlap[peaks])[::-1]][:2]

    # Return the corresponding frequencies
    omega_s_best = omg[peak_indices]
    omega_i_best = omega_p - omega_s_best   # Energy conservation
    return omega_s_best, omega_i_best


def JSA(Ei,Es,Gs,del_k,omg,omg0,phi,wp,flag):

    Es_combined = Es[0] + Es[1]
    Ei_combined = Ei[0] + Ei[1]
    omega_s_peaks, omega_i_peaks = find_intersection_frequencies(Es_combined, Ei_combined, omg, bbo_out['omega_p'])

    print(f"\n--- Peak Frequencies ---")
    print(f"Signal Peak 1: {omega_s_peaks[0]:.4e} rad/s  (Δ = {(omega_s_peaks[0] - omg0)/1e12:.3f} THz)")
    print(f"Idler Peak 1 : {omega_i_peaks[0]:.4e} rad/s  (Δ = {(omega_i_peaks[0] - omg0)/1e12:.3f} THz)")
    print(f"Sum 1        : {omega_s_peaks[0]+omega_i_peaks[0]:.4e} = ω_p = {wp:.4e}")

    print(f"Signal Peak 2: {omega_s_peaks[1]:.4e} rad/s  (Δ = {(omega_s_peaks[1] - omg0)/1e12:.3f} THz)")
    print(f"Idler Peak 2 : {omega_i_peaks[1]:.4e} rad/s  (Δ = {(omega_i_peaks[1] - omg0)/1e12:.3f} THz)")
    print(f"Sum 2        : {omega_s_peaks[1]+omega_i_peaks[1]:.4e} = ω_p = {wp:.4e}\n")



    G=np.sum((Gs)**2,axis=(-2,-1))
    G/=np.sum(G)+1e-30
    dw=(omg-omg0)/1e12
    mu = np.sum(G * dw)
    sigma = (np.sqrt(np.sum(G * (dw - mu) ** 2)/(np.sum(G) + 1e-30)))

    ws = dw[:, None]
    wi = dw[None, :]

    gamma=0.00193
    A = ((0.7)/(sigma*np.sqrt(2*gamma)))
    B = -A

    J1 = np.zeros((len(omg), len(omg)), dtype=complex)
    local_sigma = 0.05 
    gamma = 0.00193


    J = np.zeros((len(omg), len(omg)), dtype=complex)
    for ws_peak, wi_peak in zip(omega_s_peaks, omega_i_peaks):
      ws_peak_dw = (ws_peak - omg0)/1e12
      wi_peak_dw = (wi_peak - omg0)/1e12

      ws_shifted = ws - ws_peak_dw
      wi_shifted = wi - wi_peak_dw
      
      rho_k1 = np.exp(-((ws_shifted + wi_shifted) ** 2)/(2 * sigma ** 2))
      phi_k1 = np.exp(-gamma * (A * ws_shifted + B * wi_shifted) ** 2)


      J1 += rho_k1 * phi_k1
  

    J = J1 / (np.sqrt(np.sum(np.abs(J1)**2))+1e-30)

    u = Gs[1] / (np.sqrt(np.sum(np.abs(Gs[0])**2))+1e-30)
    Es_col=np.sum(Es*np.conj(u),axis=(-2,-1))
    Ei_col=np.sum(Ei*np.conj(u),axis=(-2,-1))
    Epair=np.outer(Es_col,Ei_col)
    Epair /= np.max(np.abs(Epair)) + 1e-30

    JSI = np.abs(J) ** 2
    JSI /= np.max(JSI) + 1e-30
    JSIplot = np.log10(1+500*JSI)

    plt.figure(figsize=(7,7))
    plt.xlabel("w_1 * 10^12 ")
    plt.ylabel("w_2 * 10^12 ")
    plt.imshow(JSIplot, origin='lower', cmap='turbo', interpolation='bicubic',
               aspect='equal', extent=[dw.min(),dw.max(),dw.min(),dw.max()])
    plt.colorbar(label='compressed intensity')
    plt.tight_layout()
    plt.show()
    return J, JSI, omega_s_peaks, omega_i_peaks

def HOM(J, omg, Omega, l=2,ntau=500, tau_max=3e-12):

    tau=np.linspace(-tau_max,tau_max,ntau)

    ws=omg[:,None]
    wi=omg[None,:]

    J=J/(np.sqrt(np.sum(np.abs(J)**2))+1e-30)

    C = []

    tau_c = 1e-12

    for t in tau:
      phase = np.exp(1j * (ws - wi) * t)
      integrand = J * np.conj(J.T) * phase
      P = 1 - np.real(np.sum(integrand))
      C.append(P)
    C = np.array(C)
    C = C - np.min(C)          # shift so dip bottom sits at 0
    C = C / (np.max(C) + 1e-30)
   

    plt.figure(figsize=(7,4))

    plt.plot(
        tau*1e12,
        C
    )

    plt.xlabel("Delay (ps)")
    plt.ylabel("Coincidence probability")
    plt.title("HOM Dip")

    plt.grid()

    plt.show()

    return tau,C

#######################################################################################################----main function-----------------------------------------------------------------------------------------------------
N = 100
window =3 * 0.001
w0 = 0.2* 0.001
G, X, Y = gaussian(N, window, w0)
########################################################################################################################
l = 0.002
lam = 0.81
lamp=405
wp=740
cut_theta=45

bbo_out = bbo(405, cut_theta, l)

# 41.74455302

print(bbo_out)
########################################################################################################################

steps = 100
dl = l / steps

########################################################################
Nw=200
dw=2.5e13

omg0=bbo_out['omega_s']

omg=np.linspace(omg0-dw,omg0+dw,Nw)

sigma_omg=1e12

pump_spec=np.exp(-((omg-omg0)**2) /(2*sigma_omg**2))

pump_spec=pump_spec/(np.max(pump_spec)+1e-20)

Gs=pump_spec[:,None,None]*G[None,:,:]

Es_out = np.zeros((Nw,N, N), dtype=complex)
Ei_out = np.zeros((Nw, N, N), dtype=complex)

bbo_temp = bbo_out.copy()

############################################################################################
Es_out = np.zeros((Nw, N, N), dtype=complex)
Ei_out = np.zeros((Nw, N, N), dtype=complex)

Es_vac =( np.random.randn(Nw,N,N) + 1j*np.random.randn(Nw,N,N))/1e15

Ei_vac = Es_vac.conj().transpose(0,2,1)
#############################################################################################
Ei_out_temp, Es_out_temp, phi, del_k= ssf(Gs, Es_vac, Ei_vac, Es_out, Ei_out, bbo_temp, steps, X, omg,cut_theta)

I_s_k = np.zeros((N, N))
I_i_k = np.zeros((N, N))

for w in range(Nw):

    Es_k = np.fft.fftshift( np.fft.fft2(Es_out_temp[w]))
    Ei_k = np.fft.fftshift(np.fft.fft2(Ei_out_temp[w]))
    I_s_k += np.abs(Es_k)**2
    I_i_k += np.abs(Ei_k)**2

# normalize
I_s_k /= np.max(I_s_k)
I_i_k /= np.max(I_i_k)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(np.log10(I_s_k +1e-30),cmap='Reds',vmin=-3, origin='lower')
plt.title("Signal cone")
plt.colorbar()


plt.subplot(1,3,2)
plt.imshow(np.log10(I_i_k +1e-30),cmap='Blues',vmin=-3, origin='lower')

plt.title("Idler cone")
plt.colorbar()

plt.subplot(1,3,3)

# red = signal
plt.imshow(np.log10(I_s_k +1e-30) ,cmap='Reds',alpha=0.5, vmin=-3, origin='lower')

# blue = idler
plt.imshow(
np.log10(I_i_k +1e-30) ,cmap='Blues',alpha=0.3, vmin=-3, origin='lower')

plt.title("Signal + Idler overlap")

plt.tight_layout()


S = I_s_k
I = I_i_k




plt.figure(figsize=(8,8))
plt.xlabel("k_x")
plt.ylabel("k_y")
# background
plt.imshow( np.ones_like(S),cmap='Greys',vmin=0,vmax=1, origin='lower')
level=np.array([0.4,0.5,0.6,0.7])

# signal contours
plt.contour(S,levels=level,cmap='Reds',linewidths=1,vmin=0.3)

# idler contours
plt.contour(I,levels=level,cmap='Blues',linewidths=1,vmin=0.3)
plt.colorbar()

plt.gca().set_aspect('equal')
plt.title('Signal (red) and Idler (blue) Contours')

print("Es_out_temp:", Es_out_temp.shape)
print("Ei_out_temp:", Ei_out_temp.shape)

I_s_total = np.sum(np.abs(Es_out_temp)**2, axis=0)
I_i_total = np.sum(np.abs(Ei_out_temp)**2, axis=0)


theta=np.radians(bbo_out['cut_angle_deg'])
c_hat=np.array([np.sin(theta),0,np.cos(theta)])

Es_j = np.stack([Es_out_temp,Es_out_temp])

Ei_j = np.stack([Ei_out_temp,Ei_out_temp])



Es_j = qwp(45, Es_j)
Ei_j = qwp(45, Ei_j)

# dt = 0.01
q = 5
wQ = 6e12

Es_j = linear_to_spin(Es_j)
Ei_j = linear_to_spin(Ei_j)
mid=Nw//2
phi = np.sinc(del_k[mid] * bbo_out['l_crystal'] / (2*np.pi))
print(phi.shape)
J_before, JBI, omega_s_peaks_before, omega_i_peaks_before = JSA(Ei_j,Es_j,Gs,del_k,omg,omg0,phi,wp,flag=1)

print(f"\n--- Captured Peak Frequencies (Before) ---")
print(f"Signal Peaks: {omega_s_peaks_before}")
print(f"Idler Peaks: {omega_i_peaks_before}")

t = 1
q = 2
plt.show()
print("Es_j shape:", Es_j.shape)
print("Ei_j shape:", Ei_j.shape)
############################################################
Nt=256

t=2*10e-12

t_interact = 1 / dw

q = 2  # l = 2*q = 4

Omega1 = 2e12   # ~5 bin shift for l=4 on your current grid
Omega2 = 4e12   # ~10 bin shift for l=4

Es_spin1 = q_plate(Es_j, q, omg, Omega1, X, Y)
Ei_spin1 = q_plate(Ei_j, q, omg, Omega1, X, Y)

Es_spin2 = q_plate(Es_j, q, omg, Omega2, X, Y)
Ei_spin2 = q_plate(Ei_j, q, omg, Omega2, X, Y)

Es_lin = spin_to_linear(Es_spin2)
Ei_lin = spin_to_linear(Ei_spin2)

Es_qwp = qwp(45, Es_lin)
Ei_qwp = qwp(45, Ei_lin)

Es_pol = polarizer(1, Es_qwp)
Ei_pol = polarizer(2, Ei_qwp)

phi = np.sinc(del_k[mid] * bbo_out['l_crystal'] / (2*np.pi))


mid = N//2

Es_spec = Es_spin1[0,:,mid,mid]

phase = np.unwrap(np.angle(Es_spec))

plt.figure(figsize=(6,3))

plt.plot((omg-omg0)/1e12,phase)

plt.xlabel("Δω (THz)")
plt.ylabel("phase (rad)")
plt.title("spectral phase at beam center")
J_after1, JAI, omega_s_peaks_after, omega_i_peaks_after = JSA(Ei_spin1,Es_spin1,Gs,del_k,omg,omg0,phi,wp,flag=1)

J_after2, JAI, omega_s_peaks_after, omega_i_peaks_after = JSA(Ei_spin2,Es_spin2,Gs,del_k,omg,omg0,phi,wp,flag=1)

print(f"\n--- Captured Peak Frequencies (After) ---")
print(f"Signal Peaks: {omega_s_peaks_after}")
print(f"Idler Peaks: {omega_i_peaks_after}")

tau,C1=HOM(J_before,omg,0)
_,C2=HOM(J_after1,omg,2e12)
_,C3=HOM(J_after2,omg,4e12)
plt.xlabel("Delay (ps)")
plt.ylabel("Coincidence probability")
plt.title("HOM Dip")
plt.plot(tau*1e12,C1,label='before q plate')
plt.plot(tau*1e12,C2,label='2 Trad/sec')
plt.plot(tau*1e12,C3,label='4 Trad/sec')
plt.legend()
plt.show()
print("Es_spin:", Es_spin1.shape)
print("Ei_spin:", Ei_spin1.shape)

###############plotting################################################


plt.figure(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(Es_out_temp[Nw//2])))**2 + 1e-50), cmap='turbo', origin='lower')
plt.subplot(2, 2, 2)
plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(Es_vac[Nw//2])))**2 + 1e-50), cmap='turbo', origin='lower')
plt.subplot(2, 2, 3)
plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(Ei_out_temp[Nw//2])))**2 + 1e-50), cmap='turbo', origin='lower')
plt.subplot(2, 2, 4)
plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(Ei_vac[Nw//2])))**2 + 1e-50), cmap='turbo', origin='lower')

plt.show()
