# 简历项目描述 — 基于NIRS的纺织物分类

## 中文版（简洁，适合简历项目栏）

> **基于近红外光谱的纺织纤维分类数据集构建**
>
> - 使用 **TI DLP NIRScan nano** 光谱仪（900–1700 nm, 228波段）采集 **6类** 纯纺织纤维（羊毛、腈纶、涤纶、尼龙、棉、醋酯）共 **188条** 近红外吸光度谱线
> - 样品来源涵盖 **JAMES多纤维标布** 与淘宝布料小样；对三大主要类别分别采集多个布样（涤纶24块、其余各4块），每块布样在**不同位置、旋转角度、折叠方式**下多次测量，增强数据多样性
> - 额外采集 **40条混纺光谱**（4种配比：棉/涤55:45、棉/涤80:20、尼龙/涤30:70、羊毛/涤35:65）与 **10条PVC背景光谱**
> - **后续工作**：基于该数据集开展特征波长选择研究，探索将228维光谱压缩至5–10个关键波段，为低成本便携式多光谱传感器的波段选型提供依据；同时研究纯纺模型向混纺样本的泛化能力

---

## 中文 bullet 版（更精简，直接贴简历）

> **基于NIRS的纺织纤维分类数据集**
>
> - 使用 TI DLP NIRScan nano 光谱仪（900–1700 nm, 228波段）搭建数据采集流程，采集羊毛、腈纶、涤纶、尼龙、棉、醋酯 6 类纯纺织纤维共 188 条近红外光谱
> - 样品来自 JAMES 多纤维标布与淘宝小样；通过多布样、多位置、多角度、多折叠的测量策略丰富数据多样性
> - 同步采集 40 条混纺光谱（4种配比）与 10 条背景光谱，为后续模型泛化评估提供基础
> - **后续计划**：开展特征波长选择研究，将高维光谱压缩至数个关键波段，目标是将分类模型部署于低成本便携硬件；探索纯纺训练模型对混纺样本的分类能力

---

## 英文版

> **NIR Spectroscopy Dataset for Textile Fiber Classification**
>
> - Built a near-infrared spectral dataset using a **TI DLP NIRScan nano** spectrometer (900–1700 nm, 228 bands), comprising **188 absorbance spectra** across **6 pure textile fiber classes**: Wool, Acrylic, Polyester, Nylon, Cotton, and Acetate
> - Sourced samples from JAMES multi-fiber standard fabrics and commercial fabric swatches; designed a multi-condition measurement protocol covering different swatches, positions, rotation angles, and fold configurations to capture intra-class variability
> - Collected an additional **40 blend-fabric spectra** (4 blend ratios) and **10 background spectra** as hold-out sets for generalization testing
> - **Future work**: Investigate wavelength selection methods to compress 228 spectral bands into a small subset of key wavelengths, enabling low-cost portable sensor deployment; explore cross-domain generalization from pure-fiber models to blend-fabric classification
