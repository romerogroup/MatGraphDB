import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from matgraphdb.database.utils import GLOBAL_PROP_FILE, MP_DIR

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
                  'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'][1:]

JSON_FILE = GLOBAL_PROP_FILE
SAVE_DIR= os.path.join(MP_DIR,'images')
os.makedirs(SAVE_DIR,exist_ok=True)

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


with open(JSON_FILE, 'r') as f:
    data = json.load(f)

bond_orders_std = np.array(data['bond_orders_std'])
bond_orders_avg = np.array(data['bond_orders_avg'])
n_bond_orders = np.array(data['n_bond_orders'])

# Plotting the bond_orders matrix
fig, ax = plt.subplots(figsize=(16,16))
im = ax.imshow(bond_orders_avg, cmap='hot', interpolation='nearest')
fig.colorbar(im)
ax.set_title('Bond Orders Average')
ax.set_xlabel('Atomic Symbols')
ax.set_ylabel('Atomic Symbols')
ax.set_xticks(np.arange(len(atomic_symbols)))
ax.set_yticks(np.arange(len(atomic_symbols)))
ax.set_xticklabels(atomic_symbols,rotation=90)
ax.set_yticklabels(atomic_symbols)
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
plt.savefig(os.path.join(SAVE_DIR,'bond_orders_avg.png'))  # Save the figure
plt.close()

# Plotting the bond_orders matrix
fig, ax = plt.subplots(figsize=(16,16))
im = ax.imshow(bond_orders_std, cmap='hot', interpolation='nearest')
fig.colorbar(im)
ax.set_title('Bond Orders Standard Deviation')
ax.set_xlabel('Atomic Symbols')
ax.set_ylabel('Atomic Symbols')
ax.set_xticks(np.arange(len(atomic_symbols)))
ax.set_yticks(np.arange(len(atomic_symbols)))
ax.set_xticklabels(atomic_symbols,rotation=90)
ax.set_yticklabels(atomic_symbols)
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
plt.savefig(os.path.join(SAVE_DIR,'bond_orders_std.png'))  # Save the figure
plt.close()

# Plotting the bond_orders matrix
fig, ax = plt.subplots(figsize=(16,16))
im = ax.imshow(n_bond_orders, cmap='hot', interpolation='nearest')
fig.colorbar(im)
ax.set_title('Bond Orders Occurrences')
ax.set_xlabel('Atomic Symbols')
ax.set_ylabel('Atomic Symbols')
ax.set_xticks(np.arange(len(atomic_symbols)))
ax.set_yticks(np.arange(len(atomic_symbols)))
ax.set_xticklabels(atomic_symbols,rotation=90)
ax.set_yticklabels(atomic_symbols)
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
plt.savefig(os.path.join(SAVE_DIR,'bond_orders_occurrences.png'))  # Save the figure
plt.close()


# Replace any zeros with a small number to avoid log(0)
min_nonzero = np.min(n_bond_orders[n_bond_orders > 0])
n_bond_orders_log = np.where(n_bond_orders > 0, n_bond_orders, min_nonzero)

# Plotting the bond_orders matrix with logarithmic scale
fig, ax = plt.subplots(figsize=(16, 16))
im = ax.imshow(n_bond_orders_log, cmap='hot', interpolation='nearest', norm=mcolors.LogNorm())
fig.colorbar(im)
ax.set_title('Bond Orders Occurrences')
ax.set_xlabel('Atomic Symbols')
ax.set_ylabel('Atomic Symbols')
ax.set_xticks(np.arange(len(atomic_symbols)))
ax.set_yticks(np.arange(len(atomic_symbols)))
ax.set_xticklabels(atomic_symbols, rotation=90)
ax.set_yticklabels(atomic_symbols)
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
plt.savefig(os.path.join(SAVE_DIR, 'bond_orders_occurrences_log.png'))  # Save the figure
plt.close()