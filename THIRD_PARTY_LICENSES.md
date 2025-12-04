# Third-Party Licenses

This project includes code adapted from several third-party open-source projects. The main `norfair` Rust project is licensed under the [BSD 3-Clause License](LICENSE), but the following internal modules retain their original licenses.

For complete license texts, see the individual LICENSE files linked below.

---

## Internal Modules

### **filterpy**
- **Purpose:** Kalman filtering implementation (ported from Python's filterpy library)
- **License:** MIT License
- **Copyright:** Copyright (c) 2015 Roger R. Labbe Jr
- **License File:** [src/internal/filterpy/LICENSE](src/internal/filterpy/LICENSE)

### **numpy**
- **Purpose:** NumPy-like utilities and array operations
- **License:** BSD 3-Clause License
- **Copyright:** Copyright (c) 2005-2025, NumPy Developers
- **License File:** [src/internal/numpy/LICENSE](src/internal/numpy/LICENSE)

### **scipy**
- **Purpose:** Distance metrics and spatial algorithms (ported from SciPy)
- **License:** BSD 3-Clause License
- **Copyright:** Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers
- **License File:** [src/internal/scipy/LICENSE](src/internal/scipy/LICENSE)

### **motmetrics**
- **Purpose:** MOTChallenge evaluation metrics
- **License:** MIT License
- **Copyright:** Copyright (c) 2017-2020 Christoph Heindl, Toka, Jack Valmadre
- **License File:** [src/internal/motmetrics/LICENSE](src/internal/motmetrics/LICENSE)

### **imaging**
- **Purpose:** Color palettes and constants from PIL/Pillow, Matplotlib, and Seaborn
- **License:** Multiple (MIT-CMU, Matplotlib License, BSD 3-Clause)
- **Copyright:** Various (see license file for details)
- **License File:** [src/drawing/colors/LICENSE](src/drawing/colors/LICENSE)

---

All internal modules are used in compliance with their respective open-source licenses.
