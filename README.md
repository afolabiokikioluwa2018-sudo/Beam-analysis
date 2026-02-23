# 🏗️ Beam & Frame Analysis Web Application

**CEG 410 Group 8 - Structural Analysis Project**

A complete, production-ready web application for analyzing 2D beams and frames using the Direct Stiffness Method and Slope-Deflection equations, with comprehensive reinforcement design to BS 8110:1997.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://beam-analysis-bsiiiegbidfhte7dj2fg2v.streamlit.app/)

## ✨ Features

### Core Structural Analysis Capabilities

- ✅ **Unlimited Spans**: Analyze continuous beams with any number of spans (3, 5, 10, 20+ spans)
- ✅ **Multi-Storey Frames**: 2D frame analysis for portal frames and multi-storey structures
- ✅ **Multiple Support Types**: Fixed, Pinned (Hinged), and Roller supports - any combination
- ✅ **Comprehensive Loading**:
  - Uniformly Distributed Load (UDL)
  - Varying Distributed Load (VDL/Trapezoidal)
  - Point loads (vertical & horizontal)
  - Applied moments
  - Multiple loads per member
  - Composite load combinations
- ✅ **Support Settlements**: Vertical, horizontal displacement, and rotation at any support
- ✅ **Complete Analysis Output**:
  - Support reactions at all supports
  - Member end forces with axial, shear, and moment
  - Fixed end moments (FEM) for all load types
  - Detailed shear force values (every 10% along span)
  - Detailed bending moment values (every 10% along span)
  - Interactive Shear Force Diagrams (SFD)
  - Interactive Bending Moment Diagrams (BMD)
  - Deflection diagrams with exaggerated scale
  - Maximum/minimum force locations

### Reinforcement Design (BS 8110:1997)

- ✅ **Flexural Design**: 
  - Singly reinforced beam design
  - Doubly reinforced beam detection and design
  - K and K' calculations with proper safety factors
- ✅ **Bar Arrangements**: Practical suggestions (12mm to 32mm bars)
- ✅ **Shear Design**: 
  - Shear stress calculations
  - Link spacing determination
  - BS 8110 Table 3.8 concrete shear strength
- ✅ **Deflection Checks**:
  - Span/depth ratio verification (BS 8110 Table 3.9)
  - Modification factors for steel stress
  - Actual deflection comparison against limits (Span/250, Span/350)
- ✅ **Code Compliance**: Min/max steel requirements per BS 8110

### Technical Features

- **Analysis Method**: Direct Stiffness Method (Matrix Analysis)
- **Theory**: Slope-Deflection formulation with Fixed End Moments
- **Computation**: NumPy matrix operations with SciPy linear solver
- **Visualization**: Interactive Plotly charts with zoom, pan, and hover values
- **UI**: Modern Streamlit interface with real-time updates
- **Performance**: Handles 50+ spans efficiently

## 🎯 How to Use

### Quick Start with Examples

1. **Launch the app** (see installation above)
2. **Sidebar → Quick Start Examples** → Select an example
3. Click **"Load Example"**
4. Switch to **"🔧 Analysis"** tab
5. Click **"🚀 Run Analysis"**
6. View results in **"📊 Results"**, **"📈 Diagrams"**, and **"🏗️ Reinforcement"** tabs

### Manual Structure Definition

#### Step 1: Define Nodes
- Tab: **"📐 Structure Definition"**
- Add nodes with X, Y coordinates (meters)
- Example: Node 1 at (0, 0), Node 2 at (6, 0)

#### Step 2: Define Members
- Connect nodes to create beams/columns
- Specify material properties:
  - **E**: Young's Modulus (kN/m²) - typically 200×10⁶ for steel, 30×10⁶ for concrete
  - **I**: Moment of Inertia (m⁴) - depends on cross-section
  - **A**: Cross-sectional Area (m²)

#### Step 3: Define Supports
- Add boundary conditions at nodes:
  - **Fixed**: Restrains Dx, Dy, Rotation (cantilever ends, frame bases)
  - **Pinned**: Restrains Dx, Dy (simple supports, frame bases)
  - **Roller**: Restrains Dy only (interior supports, allows horizontal movement)
- Enter settlements if applicable (Dx, Dy, Rotation in meters/radians)

#### Step 4: Define Loads
- Apply loads to members:
  - **UDL**: W1 = load intensity (kN/m)
  - **VDL**: W1 = start, W2 = end (kN/m)
  - **Point Load**: P = force (kN), a = distance from start (m)
  - **Moment**: M = moment (kNm), a = distance from start (m)
- **Multiple loads on same member?** Add multiple entries with same member number!

#### Step 5: Run Analysis
- Switch to **"🔧 Analysis"** tab
- Review structure preview
- Click **"🚀 Run Analysis"**
- Wait for success message

#### Step 6: View Results
- **"📊 Results"** tab: 
  - Reactions and displacements
  - Member end forces
  - Detailed SF & BM at span ends
  - Values at 10% intervals
  - Summary of max/min values
- **"📈 Diagrams"** tab: Interactive SFD, BMD, Deflection
- **"🏗️ Reinforcement"** tab: BS 8110 design with deflection checks

## 📊 Example Problems

### Example 1: Simply Supported Beam
```
Span: 6m
Load: 10 kN/m UDL
Supports: Pinned at left, Roller at right
Expected: Max moment at mid-span ≈ 45 kNm
```

### Example 2: Continuous Beam (3 Spans)
```
Configuration: 3 spans (4m, 4m, 4m)
Loads: Various UDL and point loads
Supports: Pinned and rollers
Features: Demonstrates moment redistribution
```

### Example 3: Portal Frame
```
Type: Simple portal
Dimensions: 6m span, 4m height
Loads: Horizontal and vertical
Supports: Fixed at base
```

### Example 4: Multi-Storey Frame
```
Type: 2-storey frame
Members: Columns and beams
Shows: Vertical and horizontal member analysis
```

### Example 5: Settlement Case
```
Configuration: 2-span beam
Settlement: 20mm at middle support
Shows: Effects of support movement on forces
```

## 🔧 Advanced Usage

### Multi-Span Beams (5+ Spans)

**For a 5-span continuous beam:**
```
Nodes: 6 (at X = 0, 5, 10, 15, 20, 25)
Members: 5 (connecting consecutive nodes)
Supports: 6 (1 Pinned, 5 Rollers)
Loads: Apply to each member as needed
```

**The app handles unlimited spans!** Just add more nodes and members.

### Portal Frames

**Setup:**
```
Nodes: 4 corners (base-left, top-left, top-right, base-right)
Members: 3 (left column, beam, right column)
Supports: 2 Fixed (at bases)
Loads: UDL on beam + lateral loads
```

### Multiple Loads on Same Member

To apply multiple loads to one member, add multiple load entries:
```
Member | Type       | Values
-------|------------|--------
2      | UDL        | W1=10
2      | Point Load | P=50, a=2
2      | Point Load | P=30, a=4
2      | Moment     | M=20, a=3
```

## 📐 Sign Conventions

- **Moments**: Sagging = Positive (bottom tension), Hogging = Negative (top tension)
- **Shear**: Positive upward on left face of section
- **Displacements**: Positive in +X, +Y directions
- **Axial Forces**: Tension positive, Compression negative

## 🏗️ Reinforcement Design (BS 8110:1997)

### Design Parameters

**Concrete Grades (fcu):**
- 25, 30, 35, 40 N/mm² (cube strength)

**Steel Grades (fy):**
- 250 N/mm² (mild steel)
- 460 N/mm² (high yield steel)

**Partial Safety Factors:**
- γm = 1.5 for concrete
- γm = 1.05 for steel

### Design Checks Performed

1. **Flexural Capacity**:
   - K = M/(bd²fcu) calculation
   - Comparison with K' (0.156 for fy=460, 0.207 for fy=250)
   - Singly or doubly reinforced determination

2. **Steel Requirements**:
   - Tension steel: As = M/(0.87 × fy × z)
   - Minimum steel: 0.13% bh (fy=460) or 0.24% bh (fy=250)
   - Maximum steel: 4% of gross area
   - Compression steel (if required)

3. **Shear Design**:
   - Shear stress: v = V/(bd)
   - Concrete shear strength from BS 8110 Table 3.8
   - Link spacing calculation if v > vc

4. **Deflection Criteria**:
   - **Span/depth ratio**: Compared against BS 8110 Table 3.9 (with modification factors)
   - **Actual deflection**: Compared against Span/250 (general) and Span/350 (brittle finishes)

### Bar Arrangements

The app suggests practical arrangements:
- 12mm, 16mm, 20mm, 25mm, 32mm diameter bars
- 2 to 6 bars per section
- Shows provided area and utilization percentage

## 🐛 Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError`
```bash
# Solution: Ensure virtual environment is activated
# Then reinstall:
pip install -r requirements.txt
```

**Problem**: App won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Try:
python -m streamlit run main.py
```

**Problem**: Port already in use
```bash
# Use different port:
streamlit run main.py --server.port 8502
```

**Problem**: Analysis fails
- Check all nodes are numbered correctly (start from 1)
- Verify members reference existing nodes
- Ensure at least one support is defined
- Check load values and units (kN, kNm, m)

**Problem**: "Singular matrix" error
- Structure is unstable (insufficient supports)
- Check support configuration
- Ensure proper boundary conditions

**Problem**: Results seem wrong
- Verify load values and units
- Check member orientations (Node_I to Node_J)
- Review support types (Fixed vs Pinned vs Roller)
- Compare with hand calculations


### Deployment Settings

- **Main file**: `main.py`
- **Python version**: 3.9+ (auto-detected)
- **Resources**: 1GB RAM (sufficient for most analyses)

## 📖 References

### Textbooks
1. **Hibbeler, R.C.** - "Structural Analysis", 10th Edition
2. **BS 8110:1997** - Structural Use of Concrete (British Standard)
3. **McGuire, Gallagher, Ziemian** - "Matrix Structural Analysis", 2nd Edition
4. **Kassimali, A.** - "Structural Analysis", 5th Edition
5. **Mosley, Bungey & Hulse** - "Reinforced Concrete Design to BS 8110", 7th Edition
6. **Ghali, A., Neville, A.M.** - "Structural Analysis: A Unified Classical and Matrix Approach"

### Design Standards
- **BS 8110-1:1997** - Structural use of concrete. Code of practice for design and construction
- **BS 8110-2:1985** - Structural use of concrete. Code of practice for special circumstances

### Key BS 8110 Clauses Implemented
- **Clause 3.4.4.4** - Design formulae for flexure
- **Clause 3.4.6** - Deflection (span/depth ratios)
- **Clause 3.12.5.3** - Minimum areas of reinforcement
- **Clause 3.12.6.1** - Maximum areas of reinforcement  
- **Clause 3.12.7** - Shear reinforcement
- **Table 3.1** - Deflection limits
- **Table 3.8** - Form and area of shear reinforcement
- **Table 3.9** - Basic span/effective depth ratios

## 📝 Units

| Quantity | Unit | Notes |
|----------|------|-------|
| Length | meters (m) | Coordinates, spans |
| Force | kilonewtons (kN) | Point loads, reactions |
| Distributed Load | kN/m | UDL, VDL |
| Moment | kilonewton-meters (kNm) | Applied moments, bending moments |
| Stress | N/mm² or kN/m² | Material properties |
| Displacement | meters (m) | Deflections, settlements |
| Steel Area | mm² | Reinforcement |
| Section Dimensions | mm | Width, depth, cover |

## 🎓 Academic Use

### CEG 410 Course Requirements

This project fulfills all requirements:
- ✅ Direct Stiffness Method implementation
- ✅ All support types (fixed, pinned, roller)
- ✅ All load types (UDL, VDL, point, moment)
- ✅ Fixed end moment calculations for all cases
- ✅ Support settlements with equivalent loads
- ✅ Complete output (reactions, forces, diagrams)
- ✅ Reinforcement design (BS 8110:1997)
- ✅ Deflection criteria verification
- ✅ Production-ready code with documentation
- ✅ Interactive visualization
- ✅ Multiple example problems

### Learning Outcomes

Students Used:
1. Matrix structural analysis methods
2. Slope-deflection theory application
3. Fixed end moment calculations
4. Support settlement effects
5. BS 8110 reinforcement design principles
6. Deflection control methods
7. Professional software development practices

### Verification Tool

Use this app to:
- Check hand calculations
- Visualize structural behavior
- Understand force distributions
- Learn design code applications
- Prepare assignment solutions

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Enhancement
- Add Eurocode design option
- Implement 3D frame analysis
- Add dynamic analysis capabilities
- Export to PDF reports
- Integration with CAD software
- Temperature load effects
- Prestressed concrete design

## 📄 License

BY THE DEVELOPER 
AFOLABI OKIKIOLUWA 

Copyright (c) 2026 Beam Analysis Project

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

## 👨‍💻 Support

For questions, issues, or feature requests:

1. Check the troubleshooting section above
2. Review example problems
3. Open an issue on GitHub
4. Afolabi Okikioluwa  (for CEG 410 students)

## 🌟 Features Comparison

| Feature | Hand Calculation | Commercial Software | This App |
|---------|-----------------|---------------------|----------|
| Unlimited spans | ❌ | ✅ | ✅ |
| Multi-storey frames | ❌ | ✅ | ✅ |
| Support settlements | ⚠️ Complex | ✅ | ✅ |
| Interactive diagrams | ❌ | ✅ | ✅ |
| BS 8110 design | ⚠️ Manual | ✅ | ✅ |
| Deflection checks | ⚠️ Manual | ✅ | ✅ |
| Free to use | ✅ | ❌ | ✅ |
| Educational | ✅ | ⚠️ Black box | ✅ |
| Open source | ✅ | ❌ | ✅ |

## 🎯 Project Statistics

- **Lines of Code**: ~3,500+
- **Functions**: 15+ analysis functions
- **Design Checks**: 10+ BS 8110 criteria
- **Examples**: 5 working examples
- **Supported Members**: Unlimited
- **Analysis Time**: < 1 second for typical problems
- **Deployment**: One-click to cloud

---

## 📞 Contact & Links

- **Live App**: [https://beam-analysis.streamlit.app](https://beam-analysis-bsiiiegbidfhte7dj2fg2v.streamlit.app/)
- **GitHub**: [github.com/afolabi okikioluwa/beam-analysis](https://github.com)
- **Documentation**: See this README
- **Issues**: GitHub Issues tab

---

**Made with ❤️ for CEG 410 Structural Analysis**
Proudly created by Okik's

*Direct Stiffness Method | Slope-Deflection | BS 8110:1997 | Professional Engineering*

---

## 🏆 Acknowledgments

- CEG 410 Course Instructors
- Afolabi Okikioluwa 
- Streamlit Development Team
- NumPy & SciPy Contributors
- Plotly Visualization Library
- Open Source Community

---

**Last Updated**: February 2026  
**Version**: 2.0.0  
**Status**: Production Ready ✅
