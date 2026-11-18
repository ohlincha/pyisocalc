#!/usr/bin/env python3
"""
isocalc_gui.py — Isotopic pattern calculator with graphical interface

Calculates isotopic distributions and simulates mass spectra for molecules
specified by chemical formula. Outputs a stick spectrum and Gaussian-broadened
envelope, with adjustable broadening to simulate instrument resolving power.

Usage:
    python3 isocalc_gui.py

Dependencies:
    Python 3.6+, numpy, matplotlib (TkAgg backend), tkinter (usually bundled
    with Python; on some Linux distributions install python3-tk separately)

License:
    GNU General Public License version 3.0 (GPLv3)

Author:
    C. André Ohlin

Isotopic data:
    Masses and natural abundances from IUPAC 1997 recommendations.
    See https://pac.iupac.org/publications/pac/pdf/1998/pdf/7001x0217.pdf

Known limitations:
    - Charge is estimated from tabulated oxidation states and will be wrong
      for many real molecules; use the charge override field.
    - Formula parser handles parentheses with integer multipliers but not
      nested parentheses or non-integer stoichiometries.

Changelog:
    0.1  (2013-01-13)  Initial release on SourceForge (Python 2.x, CLI only)
    0.2  (2013-01-xx)  Parenthesis handling added
    0.3  (2013-xx-xx)  Isotopic ratio corrections; bug fixes
    0.4  (2013-xx-xx)  Formula parser rewrite
    0.5  (2013-xx-xx)  Zero-subscript handling (K0Cl treated as Cl)
    0.8  (2013-05-30)  Last CLI-only release
    0.9  (2026-08-xx)  Port to Python 3; Tkinter GUI with embedded matplotlib;
                       algorithm corrected to per-element convolution;
                       Co/Ni ordering fixed; case-insensitive formula input;
                       interactive plot (rubber-band zoom, scroll, pan);
                       real-time broadening update; file export
"""

import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
import numpy as np

# ---------------------------------------------------------------------------
# Periodic table
# [atomic_number, oxidation_state, [isotope_masses], [isotope_abundances]]
# Isotopic data: IUPAC 1997
# ---------------------------------------------------------------------------
PERIODIC_TABLE = {
    'H':  [1,  1,  [1.0078250321, 2.0141017780],                                                       [0.999885, 0.0001157]],
    'He': [2,  0,  [3.0160293097, 4.0026032497],                                                       [0.00000137, 0.99999863]],
    'Li': [3,  1,  [6.0151233, 7.0160040],                                                             [0.0759, 0.9241]],
    'Be': [4,  2,  [9.0121821],                                                                        [1.0]],
    'B':  [5,  3,  [10.0129370, 11.0093055],                                                           [0.199, 0.801]],
    'C':  [6,  -4, [12.0, 13.0033548378],                                                              [0.9893, 0.0107]],
    'N':  [7,  5,  [14.0030740052, 15.0001088984],                                                     [0.99632, 0.00368]],
    'O':  [8,  -2, [15.9949146221, 16.99913150, 17.9991604],                                           [0.99757, 0.00038, 0.00205]],
    'F':  [9,  -1, [18.99840320],                                                                      [1.0]],
    'Ne': [10, 0,  [19.9924401759, 20.99384674, 21.99138551],                                          [0.9048, 0.0027, 0.0925]],
    'Na': [11, 1,  [22.98976967],                                                                      [1.0]],
    'Mg': [12, 2,  [23.98504190, 24.98583702, 25.98259304],                                            [0.7899, 0.10, 0.1101]],
    'Al': [13, 3,  [26.98153844],                                                                      [1.0]],
    'Si': [14, 4,  [27.9769265327, 28.97649472, 29.97377022],                                          [0.92297, 0.046832, 0.030872]],
    'P':  [15, 5,  [30.97376151],                                                                      [1.0]],
    'S':  [16, -2, [31.97207069, 32.97145850, 33.96786683, 35.96708088],                               [0.9493, 0.0076, 0.0429, 0.0002]],
    'Cl': [17, -1, [34.96885271, 36.96590260],                                                         [0.7578, 0.2422]],
    'Ar': [18, 0,  [35.96754628, 37.9627322, 39.962383123],                                            [0.003365, 0.000632, 0.996003]],
    'K':  [19, 1,  [38.9637069, 39.96399867, 40.96182597],                                             [0.932581, 0.000117, 0.067302]],
    'Ca': [20, 2,  [39.9625912, 41.9586183, 42.9587668, 43.9554811, 45.9536928, 47.952534],            [0.96941, 0.00647, 0.00135, 0.02086, 0.00004, 0.00187]],
    'Sc': [21, 3,  [44.9559102],                                                                       [1.0]],
    'Ti': [22, 4,  [45.9526295, 46.9517638, 47.9479471, 48.9478708, 49.9447921],                       [0.0825, 0.0744, 0.7372, 0.0541, 0.0518]],
    'V':  [23, 5,  [49.9471628, 50.9439637],                                                           [0.00250, 0.99750]],
    'Cr': [24, 2,  [49.9460496, 51.9405119, 52.9406538, 53.9388849],                                   [0.04345, 0.83789, 0.09501, 0.02365]],
    'Mn': [25, 2,  [54.9380496],                                                                       [1.0]],
    'Fe': [26, 3,  [53.9396148, 55.9349421, 56.9353987, 57.9332805],                                   [0.05845, 0.91754, 0.02119, 0.00282]],
    'Co': [27, 2,  [58.933195],                                                                        [1.0]],
    'Ni': [28, 2,  [57.9353479, 59.9307906, 60.9310604, 61.9283488, 63.9279696],                       [0.680769, 0.262231, 0.011399, 0.036345, 0.009256]],
    'Cu': [29, 2,  [62.9296011, 64.9277937],                                                           [0.6917, 0.3083]],
    'Zn': [30, 2,  [63.9291466, 65.9260368, 66.9271309, 67.9248476, 69.925325],                        [0.4863, 0.2790, 0.0410, 0.1875, 0.0062]],
    'Ga': [31, 3,  [68.925581, 70.9247050],                                                            [0.60108, 0.39892]],
    'Ge': [32, 2,  [69.9242504, 71.9220762, 72.9234594, 73.9211782, 75.9214027],                       [0.2084, 0.2754, 0.0773, 0.3628, 0.0761]],
    'As': [33, 3,  [74.9215964],                                                                       [1.0]],
    'Se': [34, 4,  [73.9224766, 75.9192141, 76.9199146, 77.9173095, 79.9165218, 81.9167000],           [0.0089, 0.0937, 0.0763, 0.2377, 0.4961, 0.0873]],
    'Br': [35, -1, [78.9183376, 80.916291],                                                            [0.5069, 0.4931]],
    'Kr': [36, 0,  [77.920386, 79.916378, 81.9134846, 82.914136, 83.911507, 85.9106103],               [0.0035, 0.0228, 0.1158, 0.1149, 0.5700, 0.1730]],
    'Rb': [37, 1,  [84.9117893, 86.9091835],                                                           [0.7217, 0.2783]],
    'Sr': [38, 2,  [83.913425, 85.9092624, 86.9088793, 87.9056143],                                    [0.0056, 0.0986, 0.0700, 0.8258]],
    'Y':  [39, 3,  [88.9058479],                                                                       [1.0]],
    'Zr': [40, 4,  [89.9047037, 90.9056450, 91.9050401, 93.9063158, 95.908276],                        [0.5145, 0.1122, 0.1715, 0.1738, 0.0280]],
    'Nb': [41, 5,  [92.9063775],                                                                       [1.0]],
    'Mo': [42, 6,  [91.906810, 93.9050876, 94.9058415, 95.9046789, 96.9060210, 97.9054078, 99.907477], [0.1484, 0.0925, 0.1592, 0.1668, 0.0955, 0.2413, 0.0963]],
    'Tc': [43, 2,  [96.906365, 97.907216, 98.9062546],                                                 [1.0]],
    'Ru': [44, 3,  [95.907598, 97.905287, 98.9059393, 99.9042197, 100.9055822, 101.9043495, 103.905430],[0.0554, 0.0187, 0.1276, 0.1260, 0.1706, 0.3155, 0.1862]],
    'Rh': [45, 2,  [102.905504],                                                                       [1.0]],
    'Pd': [46, 2,  [101.905608, 103.904035, 104.905084, 105.903483, 107.903894, 109.905152],            [0.0102, 0.1114, 0.2233, 0.2733, 0.2646, 0.1172]],
    'Ag': [47, 1,  [106.905093, 108.904756],                                                           [0.51839, 0.48161]],
    'Cd': [48, 2,  [105.906458, 107.904183, 109.903006, 110.904182, 111.9027572, 112.9044009, 113.9033581, 115.904755], [0.0125, 0.0089, 0.1249, 0.1280, 0.2413, 0.1222, 0.2873, 0.0749]],
    'In': [49, 3,  [112.904061, 114.903878],                                                           [0.0429, 0.9571]],
    'Sn': [50, 4,  [111.904821, 113.902782, 114.903346, 115.901744, 116.902954, 117.901606, 118.903309, 119.9021966, 121.9034401, 123.9052746], [0.0097, 0.0066, 0.0034, 0.1454, 0.0768, 0.2422, 0.0859, 0.3258, 0.0463, 0.0579]],
    'Sb': [51, 3,  [120.9038180, 122.9042157],                                                         [0.5721, 0.4279]],
    'Te': [52, 4,  [119.904020, 121.9030471, 122.9042730, 123.9028195, 124.9044247, 125.9033055, 127.9044614, 129.9062228], [0.0009, 0.0255, 0.0089, 0.0474, 0.0707, 0.1884, 0.3174, 0.3408]],
    'I':  [53, -1, [126.904468],                                                                       [1.0]],
    'Xe': [54, 0,  [123.9058958, 125.904269, 127.9035304, 128.9047795, 129.9035079, 130.9050819, 131.9041545, 133.9053945, 135.907220], [0.0009, 0.0009, 0.0192, 0.2644, 0.0408, 0.2118, 0.2689, 0.1044, 0.0887]],
    'Cs': [55, 1,  [132.905447],                                                                       [1.0]],
    'Ba': [56, 2,  [129.906310, 131.905056, 133.904503, 134.905683, 135.904570, 136.905821, 137.905241],[0.00106, 0.00101, 0.02417, 0.06592, 0.07854, 0.11232, 0.71698]],
    'La': [57, 3,  [137.907107, 138.906348],                                                           [0.00090, 0.99910]],
    'Ce': [58, 3,  [135.907140, 137.905986, 139.905434, 141.909240],                                   [0.00185, 0.00251, 0.88450, 0.11114]],
    'Pr': [59, 3,  [140.907648],                                                                       [1.0]],
    'Nd': [60, 3,  [141.907719, 142.909810, 143.910083, 144.912569, 145.913112, 147.916889, 149.920887],[0.272, 0.122, 0.238, 0.083, 0.172, 0.057, 0.056]],
    'Pm': [61, 3,  [144.91270],                                                                        [1.0]],
    'Sm': [62, 3,  [143.911995, 146.914893, 147.914818, 148.917180, 149.917271, 151.919728, 153.922205],[0.0307, 0.1499, 0.1124, 0.1382, 0.0738, 0.2675, 0.2275]],
    'Eu': [63, 3,  [150.919846, 152.921226],                                                           [0.4781, 0.5219]],
    'Gd': [64, 3,  [151.919788, 153.920862, 154.922619, 155.922120, 156.923957, 157.924101, 159.927051],[0.0020, 0.0218, 0.1480, 0.2047, 0.1565, 0.2484, 0.2186]],
    'Tb': [65, 4,  [158.925343],                                                                       [1.0]],
    'Dy': [66, 3,  [155.924278, 157.924405, 159.925194, 160.926930, 161.926795, 162.928728, 163.929171],[0.0006, 0.0010, 0.0234, 0.1891, 0.2551, 0.2490, 0.2818]],
    'Ho': [67, 3,  [164.930319],                                                                       [1.0]],
    'Er': [68, 3,  [161.928775, 163.929197, 165.930290, 166.932045, 167.932368, 169.935460],           [0.0014, 0.0161, 0.3361, 0.2293, 0.2678, 0.1493]],
    'Tm': [69, 3,  [168.934211],                                                                       [1.0]],
    'Yb': [70, 3,  [167.933894, 169.934759, 170.936322, 171.9363777, 172.9382068, 173.9388581, 175.942568], [0.0013, 0.0304, 0.1428, 0.2183, 0.1613, 0.3183, 0.1276]],
    'Lu': [71, 3,  [174.9407679, 175.9426824],                                                         [0.9741, 0.0259]],
    'Hf': [72, 4,  [173.940040, 175.9414018, 176.9432200, 177.9436977, 178.9458151, 179.9465488],      [0.0016, 0.0526, 0.1860, 0.2728, 0.1362, 0.3508]],
    'Ta': [73, 5,  [179.947466, 180.947996],                                                           [0.00012, 0.99988]],
    'W':  [74, 6,  [179.946704, 181.9482042, 182.9502230, 183.9509312, 185.9543641],                   [0.0012, 0.2650, 0.1431, 0.3064, 0.2843]],
    'Re': [75, 2,  [184.9529557, 186.9557508],                                                         [0.3740, 0.6260]],
    'Os': [76, 4,  [183.952491, 185.953838, 186.9557479, 187.9558360, 188.9581449, 189.958445, 191.961479], [0.0002, 0.0159, 0.0196, 0.1324, 0.1615, 0.2626, 0.4078]],
    'Ir': [77, 4,  [190.960591, 192.962924],                                                           [0.373, 0.627]],
    'Pt': [78, 4,  [189.959930, 191.961035, 193.962664, 194.964774, 195.964935, 197.967876],           [0.00014, 0.00782, 0.32967, 0.33832, 0.25242, 0.07163]],
    'Au': [79, 3,  [196.966552],                                                                       [1.0]],
    'Hg': [80, 2,  [195.965815, 197.966752, 198.968262, 199.968309, 200.970285, 201.970626, 203.973476],[0.0015, 0.0997, 0.1687, 0.2310, 0.1318, 0.2986, 0.0687]],
    'Tl': [81, 1,  [202.972329, 204.974412],                                                           [0.29524, 0.70476]],
    'Pb': [82, 2,  [203.973029, 205.974449, 206.975881, 207.976636],                                   [0.014, 0.241, 0.221, 0.524]],
    'Bi': [83, 3,  [208.980383],                                                                       [1.0]],
    'Po': [84, 4,  [209.0],                                                                            [1.0]],
    'At': [85, 7,  [210.0],                                                                            [1.0]],
    'Rn': [86, 0,  [220.0],                                                                            [1.0]],
    'Fr': [87, 1,  [223.0],                                                                            [1.0]],
    'Ra': [88, 2,  [226.0],                                                                            [1.0]],
    'Ac': [89, 3,  [227.0],                                                                            [1.0]],
    'Th': [90, 4,  [232.0380504],                                                                      [1.0]],
    'Pa': [91, 4,  [231.03588],                                                                        [1.0]],
    'U':  [92, 6,  [234.0409456, 235.0439231, 236.0455619, 238.0507826],                               [0.000055, 0.007200, 0.0, 0.992745]],
    'Np': [93, 5,  [237.0],                                                                            [1.0]],
    'Pu': [94, 3,  [244.0],                                                                            [1.0]],
    'Am': [95, 2,  [243.0],                                                                            [1.0]],
    'Cm': [96, 3,  [247.0],                                                                            [1.0]],
    'Bk': [97, 3,  [247.0],                                                                            [1.0]],
    'Cf': [98, 0,  [251.0],                                                                            [1.0]],
    'Es': [99, 0,  [252.0],                                                                            [1.0]],
    'Fm': [100, 0, [257.0],                                                                            [1.0]],
    'Md': [101, 0, [258.0],                                                                            [1.0]],
    'No': [102, 0, [259.0],                                                                            [1.0]],
    'Lr': [103, 0, [262.0],                                                                            [1.0]],
    'Rf': [104, 0, [261.0],                                                                            [1.0]],
    'Db': [105, 0, [262.0],                                                                            [1.0]],
    'Sg': [106, 0, [266.0],                                                                            [1.0]],
}

CUTOFF = 1e-5
MERGE_DECIMALS = 6  # rounding precision for mass binning

# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

def formula_expander(formula):
    """Expand parenthetical groups: Ca(OH)2 -> CaO2H2. Case-insensitive input."""
    while re.search(r'\(\w*\)', formula):
        parenthetical = re.findall(r'\(\w*\)[0-9]+', formula)
        for i in parenthetical:
            p = re.findall(r'[0-9]+', re.findall(r'\)[0-9]+', i)[0])
            j = re.findall(r'[A-Za-z][a-z]*[0-9]*', i)
            for n in range(len(j)):
                numero = re.findall(r'[0-9]+', j[n])
                if numero:
                    k = numero[0]
                    j[n] = re.sub(k, str(int(k) * int(p[0])), j[n])
                else:
                    j[n] = j[n] + p[0]
            formula = formula.replace(i, ''.join(j))
        if not re.findall(r'\(\w*\)[0-9]+', formula) and re.search(r'\(\w*\)', formula):
            formula = formula.replace('(', '').replace(')', '')
    for match in re.findall(r'[A-Za-z][a-z]*0', formula):
        formula = formula.replace(match, '')
    return formula


def parse_formula(formula):
    """Return dict of {element: count} from an expanded formula."""
    segs = re.findall(r'[A-Za-z][a-z]*[0-9]*', formula)
    elements = {}
    for seg in segs:
        atom_raw = re.findall(r'[A-Za-z][a-z]*', seg)[0]
        atom = atom_raw.capitalize()
        nums = re.findall(r'[0-9]+', seg)
        n = int(nums[0]) if nums else 1
        elements[atom] = elements.get(atom, 0) + n
    return elements


# ---------------------------------------------------------------------------
# Isotope pattern calculation
# ---------------------------------------------------------------------------

def _merge(masses, probs):
    """Bin masses to MERGE_DECIMALS places and sum probabilities."""
    merged = {}
    for m, p in zip(masses, probs):
        key = round(m, MERGE_DECIMALS)
        merged[key] = merged.get(key, 0.0) + p
    return list(merged.keys()), list(merged.values())


def _prune(masses, probs, cutoff=CUTOFF):
    """Remove peaks below cutoff * max."""
    maxp = max(probs)
    pairs = [(m, p) for m, p in zip(masses, probs) if p / maxp > cutoff]
    if not pairs:
        return masses, probs
    ms, ps = zip(*pairs)
    return list(ms), list(ps)


def convolve_two(masses_a, probs_a, masses_b, probs_b):
    """Convolve two isotope distributions (sum masses, multiply probs)."""
    new_m, new_p = [], []
    for ma, pa in zip(masses_a, probs_a):
        for mb, pb in zip(masses_b, probs_b):
            new_m.append(ma + mb)
            new_p.append(pa * pb)
    new_m, new_p = _merge(new_m, new_p)
    new_m, new_p = _prune(new_m, new_p)
    return new_m, new_p


def element_distribution(masses, probs, n):
    """Raise a single element's isotope distribution to the nth power."""
    result_m, result_p = [0.0], [1.0]
    for _ in range(n):
        result_m, result_p = convolve_two(result_m, result_p, masses, probs)
    return result_m, result_p


def isotope_pattern(formula):
    """
    Compute the full isotope pattern for an expanded formula string.
    Returns (masses, probs) as lists, unnormalised.
    """
    elements = parse_formula(formula)
    result_m, result_p = [0.0], [1.0]
    for atom, n in elements.items():
        em, ep = element_distribution(PERIODIC_TABLE[atom][2], PERIODIC_TABLE[atom][3], n)
        result_m, result_p = convolve_two(result_m, result_p, em, ep)
    return result_m, result_p


# ---------------------------------------------------------------------------
# Mass / charge helpers
# ---------------------------------------------------------------------------

def mol_mass(formula):
    mass = 0.0
    for atom, n in parse_formula(formula).items():
        masses = PERIODIC_TABLE[atom][2]
        abunds = PERIODIC_TABLE[atom][3]
        mass += sum(m * a for m, a in zip(masses, abunds)) * n
    return mass


def mol_charge(formula):
    charge = 0.0
    for atom, n in parse_formula(formula).items():
        charge += PERIODIC_TABLE[atom][1] * n
    return charge


def charge_label(charge):
    """Format charge as superscript-style string: 2-, 3+, etc."""
    if charge == 0:
        return ""
    sign = "-" if charge < 0 else "+"
    mag = abs(int(charge))
    return f"{mag}{sign}" if mag != 1 else sign


def title_string(formula, charge):
    return f"{formula} ({charge_label(charge)})" if charge != 0 else formula


# ---------------------------------------------------------------------------
# Final spectrum generation
# ---------------------------------------------------------------------------

def build_envelope(final, sigma=0.35, pts=1000):
    """Recompute only the Gaussian envelope from an existing peak dict."""
    keys = sorted(final.keys())
    xvec = np.linspace(keys[0] - 1, keys[-1] + 1, pts)
    yvec = np.zeros(pts)
    for mz, intensity in final.items():
        yvec += intensity * np.exp(-0.5 * ((xvec - mz) / sigma) ** 2)
    if yvec.max() > 0:
        yvec = yvec / yvec.max() * 100.0
    return xvec, yvec


def build_spectrum(formula_raw, charge_override, sigma=0.35):
    """
    Full pipeline.
    Returns (formula, avg_mass, calc_charge, used_charge, final_dict, xvec, yvec)
    where final_dict maps m/z -> relative intensity (%).
    Raises ValueError on bad input.
    """
    formula = formula_expander(formula_raw)
    if not formula:
        raise ValueError("Empty formula after expansion.")

    elements = parse_formula(formula)
    for atom in elements:
        if atom not in PERIODIC_TABLE:
            raise ValueError(f"Unknown element: {atom}")

    avg_mass = mol_mass(formula)
    calc_charge = mol_charge(formula)

    # Charge for m/z calculation: user override > auto > 1
    used_charge = charge_override if charge_override != 0 else (int(calc_charge) if calc_charge != 0 else 1)

    masses, probs = isotope_pattern(formula)

    # Normalise and convert to m/z
    maxp = max(probs)
    charge_abs = abs(used_charge)
    final = {}
    for m, p in zip(masses, probs):
        rel = p / maxp * 100.0
        if rel > 0.01:
            mz = round(m / charge_abs, 6)
            final[mz] = final.get(mz, 0.0) + rel

    if not final:
        raise ValueError("No peaks above threshold.")

    # Gaussian envelope
    xvec, yvec = build_envelope(final, sigma=sigma)

    return formula, avg_mass, calc_charge, used_charge, final, xvec, yvec


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class IsocalcApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Isotope Pattern Calculator")
        self.geometry("1000x620")
        self.resizable(True, True)
        self._last_result = None  # cache for file export
        self._build_ui()

    def _build_ui(self):
        # ---- top control strip ----
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Formula:").pack(side=tk.LEFT)
        self.formula_var = tk.StringVar()
        formula_entry = ttk.Entry(ctrl, textvariable=self.formula_var, width=20, font=("TkFixedFont", 12))
        formula_entry.pack(side=tk.LEFT, padx=(4, 12))
        formula_entry.bind("<Return>", lambda e: self._run())
        formula_entry.focus()
        self.formula_var.trace_add('write', lambda *_: self._invalidate())

        ttk.Label(ctrl, text="Charge (0 = auto):").pack(side=tk.LEFT)
        self.charge_var = tk.IntVar(value=0)
        ttk.Spinbox(ctrl, from_=-20, to=20, textvariable=self.charge_var, width=5).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Broadening (σ):").pack(side=tk.LEFT)
        self.sigma_var = tk.DoubleVar(value=0.35)
        ttk.Spinbox(ctrl, from_=0.01, to=5.0, increment=0.05,
                    textvariable=self.sigma_var, width=6, format="%.2f").pack(side=tk.LEFT, padx=(4, 12))

        # Real-time update when sigma changes (envelope only, no recalculation)
        self.sigma_var.trace_add('write', lambda *_: self._redraw_envelope())

        ttk.Button(ctrl, text="Calculate", command=self._run).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Clear",     command=self._clear).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Save…",     command=self._save).pack(side=tk.LEFT, padx=4)

        # ---- info bar ----
        self.info_var = tk.StringVar(value="Enter a formula and press Calculate or Return.")
        ttk.Label(self, textvariable=self.info_var, anchor=tk.W, padding=(8, 2)).pack(side=tk.TOP, fill=tk.X)

        # ---- main area ----
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # -- plot panel --
        plot_frame = ttk.Frame(pane)
        pane.add(plot_frame, weight=3)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._init_axes()
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Zoom/pan controls
        nav = ttk.Frame(plot_frame)
        nav.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(nav, text="-", width=3, command=lambda: self._zoom(1.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="+", width=3, command=lambda: self._zoom(1/1.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Reset", command=self._reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav, text="  Drag to zoom  |  Scroll to zoom  |  Right-drag to pan",
                  foreground='gray').pack(side=tk.LEFT, padx=8)

        # Mouse state
        self._pan_start = None
        self._band_start = None
        self._band_rect  = None
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)

        # -- peak list panel --
        table_frame = ttk.Frame(pane)
        pane.add(table_frame, weight=1)

        ttk.Label(table_frame, text="Peak list", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))

        cols = ("m/z", "Intensity (%)")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.E)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    def _init_axes(self):
        self.ax.set_xlabel("m/z")
        self.ax.set_ylabel("Relative intensity (%)")
        self.ax.set_ylim(0, 110)

    def _set_xlim(self, xmin, xmax):
        """Set x limits, clamped to the full data range, y always fixed."""
        if hasattr(self, '_xlim_full'):
            lo, hi = self._xlim_full
            span = xmax - xmin
            # clamp pan so we don't scroll off the data
            if xmin < lo:
                xmin, xmax = lo, lo + span
            if xmax > hi:
                xmin, xmax = hi - span, hi
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(0, 110)
        self.canvas.draw_idle()

    def _reset_zoom(self):
        if hasattr(self, '_xlim_full'):
            lo, hi = self._xlim_full
            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(0, 110)
            self.canvas.draw_idle()

    def _zoom(self, factor, centre=None):
        """Zoom x axis by factor around centre (data coords); >1 zooms out."""
        xmin, xmax = self.ax.get_xlim()
        if centre is None:
            centre = (xmin + xmax) / 2.0
        new_half = (xmax - xmin) * factor / 2.0
        self._set_xlim(centre - new_half, centre + new_half)

    # ---- mouse event handlers ----

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 1:
            # Left: start rubber-band zoom
            self._band_start = event.xdata
            self._band_rect = Rectangle(
                (event.xdata, 0), 0, 110,
                transform=self.ax.transData,
                facecolor='steelblue', alpha=0.15, edgecolor='steelblue', linewidth=1
            )
            self.ax.add_patch(self._band_rect)
        elif event.button == 3:
            # Right: start pan
            self._pan_start = event.xdata

    def _on_release(self, event):
        if event.button == 1 and self._band_start is not None:
            if self._band_rect is not None:
                self._band_rect.remove()
                self._band_rect = None
            if event.xdata is not None:
                x0 = min(self._band_start, event.xdata)
                x1 = max(self._band_start, event.xdata)
                if x1 - x0 > 0.01:  # ignore accidental clicks
                    self._set_xlim(x0, x1)
                else:
                    self.canvas.draw_idle()
            self._band_start = None
        elif event.button == 3:
            self._pan_start = None

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._band_start is not None and self._band_rect is not None:
            # Update rubber band width
            x0 = self._band_start
            x1 = event.xdata
            self._band_rect.set_x(min(x0, x1))
            self._band_rect.set_width(abs(x1 - x0))
            self.canvas.draw_idle()
        elif self._pan_start is not None:
            dx = self._pan_start - event.xdata
            xmin, xmax = self.ax.get_xlim()
            self._set_xlim(xmin + dx, xmax + dx)

    def _on_scroll(self, event):
        """Scroll up zooms in, scroll down zooms out."""
        if event.inaxes != self.ax:
            return
        factor = 1/1.15 if event.step > 0 else 1.15
        self._zoom(factor, centre=event.xdata)

    def _invalidate(self):
        """Null the cached result when the formula changes so stale data can't be reused."""
        self._last_result = None

    def _redraw_envelope(self):
        """Recompute and redraw only the Gaussian envelope using the cached peak dict."""
        if self._last_result is None:
            return
        try:
            sigma = max(0.01, self.sigma_var.get())
        except tk.TclError:
            return  # spinbox mid-edit
        formula, avg_mass, used_charge, final, _xvec, _yvec = self._last_result
        xvec, yvec = build_envelope(final, sigma=sigma)
        self._last_result = (formula, avg_mass, used_charge, final, xvec, yvec)
        # Redraw envelope line only (first line is stems, second is envelope)
        if len(self.ax.lines) >= 1:
            self.ax.lines[-1].set_data(xvec, yvec)
        self.ax.set_ylim(0, 110)
        self.canvas.draw_idle()

    def _run(self):
        formula_raw = self.formula_var.get().strip()
        if not formula_raw:
            return
        try:
            charge_override = self.charge_var.get()
        except tk.TclError:
            charge_override = 0

        try:
            sigma = max(0.01, self.sigma_var.get())
            formula, avg_mass, calc_charge, used_charge, final, xvec, yvec = build_spectrum(
                formula_raw, charge_override, sigma=sigma)
        except (ValueError, KeyError, RecursionError) as exc:
            messagebox.showerror("Error", str(exc))
            return

        self._last_result = (formula, avg_mass, used_charge, final, xvec, yvec)

        sign_str = charge_label(used_charge)
        self.info_var.set(
            f"Formula: {formula}    "
            f"Avg. mass: {avg_mass:.4f} Da    "
            f"Calc. charge: {int(calc_charge)}    "
            f"Used charge: {used_charge} ({sign_str})"
        )

        # Update plot
        self.ax.clear()
        # Bin sticks to 2 d.p. for display only (leaves final dict untouched)
        binned = {}
        for mz, intensity in final.items():
            key = round(mz, 2)
            binned[key] = binned.get(key, 0.0) + intensity
        binned_max = max(binned.values())
        peak_mz  = list(binned.keys())
        peak_int = [v / binned_max * 100.0 for v in binned.values()]
        self.ax.vlines(peak_mz, 0, peak_int, colors='steelblue', linewidth=1.2)
        self.ax.plot(xvec, yvec, color='tomato', linewidth=1.0, alpha=0.8)
        self._init_axes()
        self.ax.set_title(title_string(formula, used_charge))
        self.fig.tight_layout()
        self.canvas.draw()
        self._xlim_full = self.ax.get_xlim()  # store for zoom/pan clamping

        # Update peak table
        for row in self.tree.get_children():
            self.tree.delete(row)
        for mz in sorted(final):
            self.tree.insert("", tk.END, values=(f"{mz:.4f}", f"{final[mz]:.2f}"))

    def _clear(self):
        self.formula_var.set("")
        self.charge_var.set(0)
        self.info_var.set("Enter a formula and press Calculate or Return.")
        self.ax.clear()
        self._init_axes()
        self.fig.tight_layout()
        self.canvas.draw()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._last_result = None

    def _save(self):
        if self._last_result is None:
            messagebox.showinfo("Nothing to save", "Run a calculation first.")
            return
        formula, avg_mass, used_charge, final, xvec, yvec = self._last_result
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Tab-separated text", "*.txt"), ("All files", "*.*")],
            initialfile=f"{formula}.txt",
            title="Save peak list"
        )
        if not path:
            return
        try:
            with open(path, 'w') as f:
                f.write(f"# Isotope pattern: {title_string(formula, used_charge)}\n")
                f.write(f"# Avg. mass: {avg_mass:.6f} Da    Charge: {used_charge}\n")
                f.write("# m/z\tIntensity (%)\n")
                for mz in sorted(final):
                    f.write(f"{mz:.6f}\t{final[mz]:.4f}\n")
                f.write("\n# Gaussian envelope\n# m/z\tIntensity (%)\n")
                for x, y in zip(xvec, yvec):
                    f.write(f"{x:.6f}\t{y:.4f}\n")
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.info_var.set(self.info_var.get() + f"    [Saved: {path}]")


if __name__ == "__main__":
    app = IsocalcApp()
    app.mainloop()
