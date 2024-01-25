from itertools import combinations,combinations_with_replacement

atomic_names = ['', 'Hydrogen', 'Helium',
                'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon',
                'Sodium', 'Magnesium', 'Aluminium', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon',
                'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron', 'Cobalt',
                'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton',
                'Rubidium', 'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium',
                'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine',
                'Xenon', 'Caesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium', 'Promethium',
                'Samarium', 'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium',
                'Ytterbium', 'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium', 'Platinum',
                'Gold', 'Mercury', 'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine', 'Radon', 'Francium',
                'Radium', 'Actinium', 'Thorium', 'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium',
                'Curium', 'Berkelium', 'Californium', 'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
                'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium', 'Bohrium', 'Hassium', 'Meitnerium',
                'Darmstadtium', 'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium', 'moscovium', 'Livermorium',
                'Tennessine', 'Oganesson']

atomic_symbols = ['',
                  'H', 'He',
                  'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                  'Kr',
                  'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                  'Xe',
                  'Cs', 'Ba',
                  'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                  'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                  'Fr', 'Ra',
                  'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
                  'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

electronegativities = [None, 2.2, None, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98,
                       None, 0.93, 1.31, 1.61, 1.9, 2.19, 2.58, 3.16, None, 0.82,
                       1.0, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.9,
                       1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.0, 0.82, 0.95, 1.22,
                       1.33, 1.6, 2.16, 1.9, 2.2, 2.28, 2.2, 1.93, 1.69, 1.78,
                       1.96, 2.05, 2.1, 2.66, 2.6, 0.79, 0.89, 1.1, 1.12, 1.13,
                       1.14, 1.13, 1.17, 1.2, 1.2, 1.2, 1.22, 1.23, 1.24, 1.25,
                       1.1, 1.27, 1.3, 1.5, 2.36, 1.9, 2.2, 2.2, 2.28, 2.54,
                       2.0, 1.62, 1.87, 2.02, 2.0, 2.2, 2.2, 0.7, 0.9, 1.1,
                       1.3, 1.5, 1.38, 1.36, 1.28, 1.3, 1.3, 1.3, 1.3, 1.3,
                       1.3, 1.3, 1.3, None, None, None, None, None, None, None,
                       None, None, None, None, None, None, None, None, None]

valences_nominal = [0,
                    1, 0,
                    1, 2, 3, 4, 5, 2, 1, 0,
                    1, 2, 3, 4, 5, 6, 7, 0,
                    1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 4, 2, 3, 4, 5, 6, 7, 0,
                    1, 2, 3, 4, 5, 6, 7, 8, 6, 4, 4, 2, 3, 4, 5, 6, 7, 0,
                    1, 2,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    4, 5, 6, 7, 8, 8, 6, 5, 4, 3, 4, 5, 6, 7, 0,
                    1, 2,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    4, 5, 6, 7, 8,
                    None, None, None, None, None, None, None, None, None, None]

periods = [None, 1, 1,
           2, 2, 2, 2, 2, 2, 2, 2,
           3, 3, 3, 3, 3, 3, 3, 3,
           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

groups = [None,
          1, 18,
          1, 2, 13, 14, 15, 16, 17, 18,
          1, 2, 13, 14, 15, 16, 17, 18,
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          1, 2,
          -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          1, 2,
          -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

blocks = [None,
          's', 'p',
          's', 's', 'p', 'p', 'p', 'p', 'p', 'p',
          's', 's', 'p', 'p', 'p', 'p', 'p', 'p',
          's', 's', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'p', 'p', 'p', 'p', 'p', 'p',
          's', 's', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'p', 'p', 'p', 'p', 'p', 'p',
          's', 's',
          'd', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
          'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'p', 'p', 'p', 'p', 'p', 'p',
          's', 's',
          'd', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
          'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'p', 'p', 'p', 'p', 'p', 'p']


# covalent_radii = [0.20, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66,  # H .. O
#                   0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02,  # F .. Cl
#                   1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32,  # Ar .. Fe
#                   1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20,  # Co .. Br
#                   1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,  # Kr .. Ru
#                   1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39,  # Rh .. I
#                   1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98,  # Xe .. Sm
#                   1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,  # Eu .. Lu
#                   1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,  # Hf .. Hg
#                   1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15,  # Tl .. Ac
#                   2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.68, 1.68,  # Th .. Cf
#                   1.65, 1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41,  # Es .. Bh
#                   1.34, 1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75,  # Hs .. Lv
#                   1.65, 1.57]

covalent_radii = [None, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66,  # H .. O
                  0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02,  # F .. Cl
                  1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32,  # Ar .. Fe
                  1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20,  # Co .. Br
                  1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,  # Kr .. Ru
                  1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39,  # Rh .. I
                  1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98,  # Xe .. Sm
                  1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,  # Eu .. Lu
                  1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,  # Hf .. Hg
                  1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15,  # Tl .. Ac
                  2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.68, 1.68,  # Th .. Cf
                  1.65, 1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41,  # Es .. Bh
                  1.34, 1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75,  # Hs .. Lv
                  1.65, 1.57]


masses = [0.0, 1.00794, 4.002602, 6.941, 9.012182,   # None + H - Be
          10.811, 12.011, 14.00674, 15.9994, 18.9984032,    # B - F
          20.1797, 22.989768, 24.3050, 26.981539, 28.0855,  # Ne -Si
          30.973762, 32.066, 35.4527, 39.948, 39.0983,      # P - K
          40.078, 44.955910, 47.88, 50.9415, 51.9961,       # Ca - Cr
          54.93805, 55.847, 58.93320, 58.69, 63.546,        # Mn - Cu
          65.39, 69.723, 72.61, 74.92159, 78.96,            # Zn - Se
          79.904, 83.80, 85.4678, 87.62, 88.90585,          # Br - Y
          91.224, 92.90638, 95.94, 98.9062, 101.07,         # Zr - Ru
          102.9055, 106.42, 107.8682, 112.411, 114.82,      # Rh - In
          118.710, 121.753, 127.60, 126.90447, 131.29,      # Sn - Xe
          132.90543, 137.327, 138.9055, 140.115, 140.90765,  # Cs - Pr
          144.24, 147.91, 150.36, 151.965, 157.25,          # Nd - Gd
          158.92534, 162.50, 164.93032, 167.26, 168.93421,  # Tb - Tm
          173.04, 174.967, 178.49, 180.9479, 183.85,        # Yb - W
          186.207, 190.2, 192.22, 195.08, 196.96654,        # Re - Au
          200.59, 204.3833, 207.2, 208.98037, 209.0,        # Hg - Po
          210.0, 222.0, 223.0, 226.0254, 230.0,             # At - Ac
          232.0381, 231.0359, 238.0289, 237.0482, 242.0,    # Th - Pu
          243.0, 247.0, 247.0, 249.0, 254.0,                # Am - Es
          253.0, 256.0, 254.0, 257.0, 260.0,                # Fm - Rf
          268.0,  # Db
          269.0,  # Sg
          270.0,  # Bh
          270.0,  # Hs
          278.0,  # Mt
          281.0,  # Ds
          282.0,  # Rg
          285.0,  # Cn
          286.0,  # Nh
          290.0,  # Fl
          290.0,  # Mc
          293.0,  # Lv
          294.0,  # Ts
          294.0]  # Og
        
atomic_symbols_map={key:value for value, key in enumerate(atomic_symbols[1:])}
def covalent_cutoff_map(tol=0.1):
    cutoff_dict={}
    element_combs=list(combinations_with_replacement(atomic_symbols[1:],r=2))

    covalent_map={ element:covalent_radii for element,covalent_radii in zip(atomic_symbols[1:], covalent_radii[1:])}
    for element_comb in element_combs:
        element_1=element_comb[0]
        element_2=element_comb[1]
        covalent_radii_1=covalent_map[element_1]
        covalent_radii_2=covalent_map[element_2]
        cutoff = (covalent_radii_1 + covalent_radii_2)*(1+tol)
        cutoff_dict.update({element_comb:cutoff})

    return cutoff_dict

