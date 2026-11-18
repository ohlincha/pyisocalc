# pyisocalc

Isotopic pattern calculator with a graphical interface. Enter a chemical formula and
get the isotopic distribution as a stick spectrum plus a Gaussian-broadened envelope,
with adjustable broadening to simulate instrument resolving power.

Originally released on [SourceForge](https://sourceforge.net/projects/pyisocalc/)
in 2013 (Python 2, command line only). Version 0.9 is a Python 3 port with a Tk GUI
and an improved algorithm.

## Features

- Formula input with (non-nested) parentheses, e.g. `Ca(OH)2` or `K8(Nb6O19)`.
  Input is case-insensitive.
- Exact per-element convolution of isotope distributions (IUPAC 1997 masses and
  abundances).
- Charge estimated from typical oxidation states, or set manually.
- Interactive plot: drag to zoom, scroll to zoom, right-drag to pan.
- Broadening (σ) updates the envelope in real time.
- Peak list table, and export of peaks and envelope to a tab-separated text file.

## Installation

Requires Python 3.6+ with tkinter, numpy and matplotlib.

```sh
pip install -r requirements.txt
python3 isocalc_gui.py
```

On Debian/Ubuntu you can instead use the system packages:

```sh
sudo apt install python3-tk python3-numpy python3-matplotlib
```

## Known limitations

- The automatic charge comes from tabulated oxidation states and will be wrong for
  many real molecules. Use the charge field to override it.
- Nested parentheses and non-integer stoichiometries are not supported.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

Author: C. André Ohlin

## Older version

The original Python 2 command-line version (v0.8) is kept in [`legacy/`](legacy/).
