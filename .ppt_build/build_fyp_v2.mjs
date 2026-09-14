import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/ROG/Desktop/FYP";
const SKILL_DIR = "C:/Users/ROG/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".ppt_build", "v2_work");
const FINAL_PPTX = path.join(workspaceDir, "outputs", "FYP版本2.pptx");
const templatePath = path.join(workspaceDir, "调研汇报模版.pptx");
const leadImagePath = "C:/Users/ROG/Desktop/LEAD/figs/lead_pipeline.png";
const RUNTIME_PYTHON = "C:/Users/ROG/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";

const { makeNativeBulletParagraphs, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const deck = Presentation.create({ slideSize: { width: 960, height: 540 } });
for (let i = 0; i < 13; i += 1) {
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 74, width: 960, height: 2 },
    fill: "#2196D2",
    line: { fill: "none", width: 0 },
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 504, width: 960, height: 36 },
    fill: "#2196D2",
    line: { fill: "none", width: 0 },
  });
}

const W = 960;
const H = 540;
const BLUE = "#2196D2";
const DARK = "#1F2937";
const MID = "#56616F";
const LIGHT = "#E9F5FA";
const PALE = "#F5F9FB";
const RED = "#C53D35";
const GOLD = "#D49A22";
const GREEN = "#2B8A72";
const WHITE = "#FFFFFF";
const GRID = "#C9D7DF";
const FONT = "宋体";
const LATIN = "Arial";

function rect(slide, left, top, width, height, fill, lineFill = "none", radius = false) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: { fill: lineFill, width: lineFill === "none" ? 0 : 1 },
  });
}

function textbox(slide, text, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: opts.typeface ?? FONT,
    fontSize: opts.fontSize ?? 20,
    bold: opts.bold ?? false,
    color: opts.color ?? DARK,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "middle",
    autoFit: "shrinkText",
    ...(opts.italic ? { italic: true } : {}),
  };
  return box;
}

function bullets(slide, items, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: opts.marginLeftPoints ?? 17,
    hangingPoints: opts.hangingPoints ?? 8,
    spaceAfterPoints: opts.spaceAfterPoints ?? 7,
  });
  box.text.style = {
    typeface: opts.typeface ?? FONT,
    fontSize: opts.fontSize ?? 19,
    color: opts.color ?? DARK,
    autoFit: "shrinkText",
    verticalAlignment: "top",
  };
  return box;
}

function cleanBody(slide) {
  rect(slide, 0, 76, W, 428, WHITE);
}

function setTitle(slide, title) {
  return textbox(slide, title, 20, 12, 725, 48, {
    typeface: FONT, fontSize: 28, bold: true, color: DARK,
  });
}

function pageNo(slide, number) {
  rect(slide, 850, 504, 85, 36, BLUE);
  textbox(slide, String(number).padStart(2, "0"), 870, 507, 48, 25, {
    typeface: LATIN, fontSize: 14, color: WHITE, alignment: "right",
  });
}

function note(slide, value) {
  slide.speakerNotes.textFrame.setText(value);
}

function toc(slide, active) {
  cleanBody(slide);
  setTitle(slide, "CONTENTS");
  const items = [
    ["01", "研究背景", "公开数据上的问题与挑战"],
    ["02", "研究现状", "分类范式与证据缺口"],
    ["03", "代表方法", "LEAD与本项目框架"],
    ["04", "后续工作", "公开数据提升与外部验证"],
  ];
  items.forEach(([num, title, sub], i) => {
    const y = 120 + i * 82;
    const isActive = i === active;
    if (isActive) rect(slide, 43, y + 4, 6, 52, BLUE);
    textbox(slide, num, 62, y, 52, 58, {
      typeface: LATIN, fontSize: 17, bold: true, color: isActive ? BLUE : MID,
    });
    textbox(slide, title, 118, y, 180, 32, {
      fontSize: 22, bold: true, color: isActive ? DARK : MID,
    });
    textbox(slide, sub, 300, y, 560, 32, {
      fontSize: 17, color: isActive ? RED : MID,
    });
  });
  pageNo(slide, active === 0 ? 2 : active === 1 ? 5 : active === 2 ? 7 : 11);
}

// 1. Cover
{
  const slide = deck.slides.items[0];
  rect(slide, 0, 0, W, 504, WHITE);
  rect(slide, 55, 128, 7, 168, BLUE);
  textbox(slide, "基于 LEAD 的多疾病 EEG\n关键因素筛选", 88, 115, 760, 125, {
    fontSize: 38, bold: true, color: DARK,
  });
  textbox(slide, "公开数据集性能提升与电极、频段、特征的稳定贡献", 90, 252, 760, 48, {
    fontSize: 21, color: MID,
  });
  textbox(slide, "FYP 调研汇报", 620, 374, 250, 32, {
    fontSize: 18, bold: true, color: DARK, alignment: "right",
  });
  textbox(slide, "2026-09-13", 620, 410, 250, 26, {
    typeface: LATIN, fontSize: 16, color: MID, alignment: "right",
  });
  pageNo(slide, 1);
  note(slide, "本页使用用户提供的调研汇报模板。研究主线：公开数据集上的可复现性能提升；医院数据仅作后续外部验证。主模型为LEAD。 ");
}

toc(deck.slides.items[1], 0);

// 3. Background
{
  const slide = deck.slides.items[2];
  cleanBody(slide);
  setTitle(slide, "EEG疾病识别中的共同输入");
  textbox(slide, "多篇研究反复使用静息态EO/EC、相似导联和固定切窗", 48, 94, 864, 45, {
    fontSize: 23, bold: true,
  });
  const labels = [
    ["记录条件", "静息态 EO / EC"],
    ["导联设置", "19 / 31 / 32 / 64"],
    ["切窗长度", "2–12 s"],
    ["常用特征", "PSD、连接、复杂度"],
  ];
  labels.forEach(([a, b], i) => {
    const x = 55 + i * 222;
    textbox(slide, a, x, 182, 180, 32, { fontSize: 17, bold: true, color: BLUE, alignment: "center" });
    textbox(slide, b, x, 225, 180, 58, { fontSize: 19, bold: true, alignment: "center" });
  });
  rect(slide, 70, 334, 820, 1, BLUE);
  textbox(slide, "这些选择提供了可比较的输入条件，但不能回答哪些因素真正支撑分类。", 85, 355, 790, 48, {
    fontSize: 22, bold: true, color: RED, alignment: "center",
  });
  textbox(slide, "研究问题：哪些频段、电极和特征在不同被试划分及公开数据集上保持不可替代贡献？", 90, 421, 780, 50, {
    fontSize: 19, color: DARK, alignment: "center",
  });
  pageNo(slide, 3);
  note(slide, "来源：用户提供的EEG文章筛选.xlsx与现有FYP版本1。共同输入条件属于文献观察，不代表因果或决定性证据。 ");
}

// 4. Challenges
{
  const slide = deck.slides.items[3];
  cleanBody(slide);
  setTitle(slide, "公开数据集上的三项核心挑战");
  const xs = [52, 349, 646];
  const blocks = [
    ["数据异构", "通道、采样率、参考方式和记录状态不同，模型容易学习数据集来源。"],
    ["评价偏差", "片段随机划分会造成被试泄漏，片段准确率也不能代替患者级性能。"],
    ["解释不稳定", "单次注意力或高准确率不足以证明某个频段、电极具有稳定贡献。"],
  ];
  blocks.forEach(([head, body], i) => {
    textbox(slide, `0${i + 1}`, xs[i], 128, 60, 38, { typeface: LATIN, fontSize: 20, bold: true, color: BLUE });
    textbox(slide, head, xs[i], 170, 245, 40, { fontSize: 23, bold: true });
    textbox(slide, body, xs[i], 226, 240, 125, { fontSize: 18, color: MID, verticalAlignment: "top" });
  });
  rect(slide, 52, 388, 856, 1, GRID);
  textbox(slide, "主评测场：公开数据集", 62, 410, 270, 34, { fontSize: 20, bold: true, color: GREEN });
  textbox(slide, "医院数据放在模型定型之后，仅用于外部验证，不参与模型选择和超参数调优。", 315, 405, 570, 48, {
    fontSize: 18, bold: true, color: RED,
  });
  pageNo(slide, 4);
  note(slide, "研究范围由用户确认：医院数据仅作后续实验验证，项目重点是公开数据集上的性能提升。 ");
}

toc(deck.slides.items[4], 1);

// 6. State of the art table
{
  const slide = deck.slides.items[5];
  cleanBody(slide);
  setTitle(slide, "现有EEG疾病识别方法与证据缺口");
  const table = slide.tables.add({
    rows: 4,
    columns: 4,
    left: 45,
    top: 115,
    width: 870,
    height: 260,
    columnWidths: [145, 260, 245, 220],
    values: [
      ["方法范式", "主要表示", "优势", "与本项目的缺口"],
      ["手工特征", "PSD、相对功率、连接、熵", "含义明确，适合小样本", "依赖特征选择，跨数据集稳定性有限"],
      ["监督深度模型", "CNN、EEGNet、Conformer", "自动学习时空模式", "常在单数据集内评估，解释结果不一致"],
      ["EEG预训练模型", "LEAD、LaBraM等", "利用大规模无标签或多数据集表示", "仍需验证疾病级迁移和关键因素的可重复性"],
    ],
  });
  table.borders.assign({ style: "solid", fill: GRID, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({
    fill: "#243746",
    textStyle: { typeface: FONT, fontSize: 16, bold: true, color: WHITE },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 3, columnCount: 4 }).assign({
    textStyle: { typeface: FONT, fontSize: 15, color: DARK },
    margins: { left: 8, right: 8, top: 5, bottom: 5 },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 3, columnCount: 1 }).assign({
    fill: LIGHT,
    textStyle: { typeface: FONT, fontSize: 16, bold: true, color: DARK },
  });
  textbox(slide, "现有方法关注“能否分类”；本项目进一步检验“哪些信息稳定贡献，并带来公开数据集上的提升”。", 65, 408, 830, 54, {
    fontSize: 19, bold: true, color: RED, alignment: "center",
  });
  pageNo(slide, 6);
  note(slide, "来源：EEG文章筛选.xlsx、LEAD论文与FYP版本1中的方法综述。LaBraM仅在此作为预训练模型类别示例。 ");
}

toc(deck.slides.items[6], 2);

// 8. Motivation
{
  const slide = deck.slides.items[7];
  cleanBody(slide);
  setTitle(slide, "框架动机：从异构输入到稳定贡献");
  const rows = [
    ["问题一", "公开EEG数据的通道与采样率不同", "LEAD的通道位置与采样率嵌入"],
    ["问题二", "疾病标签属于被试，而非单个短片段", "样本级与被试级联合学习和评价"],
    ["问题三", "高性能仍不能说明关键频段与电极", "加入配对消融、频段门控和稳定性统计"],
  ];
  textbox(slide, "研究缺口", 65, 103, 330, 34, { fontSize: 18, bold: true, color: MID });
  textbox(slide, "框架响应", 545, 103, 330, 34, { fontSize: 18, bold: true, color: MID });
  rows.forEach(([tag, gap, response], i) => {
    const y = 150 + i * 92;
    textbox(slide, tag, 65, y, 82, 50, { fontSize: 17, bold: true, color: BLUE, alignment: "center" });
    textbox(slide, gap, 155, y, 325, 50, { fontSize: 18, bold: true });
    textbox(slide, "›", 490, y, 40, 50, { typeface: LATIN, fontSize: 31, bold: true, color: BLUE, alignment: "center" });
    textbox(slide, response, 545, y, 330, 50, { fontSize: 18, bold: true });
    if (i < 2) rect(slide, 65, y + 72, 810, 1, GRID);
  });
  textbox(slide, "设计原则：先建立可复现的LEAD基线，再用受控变量实验定位贡献，最后验证改进是否跨公开数据集成立。", 70, 437, 820, 46, {
    fontSize: 18, bold: true, color: RED, alignment: "center",
  });
  pageNo(slide, 8);
  note(slide, "LEAD技术依据来自本地论文2502.01678v4.pdf和C:/Users/ROG/Desktop/LEAD代码。第三项为本项目在LEAD基础上的研究扩展。 ");
}

// 9. Architecture
{
  const slide = deck.slides.items[8];
  cleanBody(slide);
  setTitle(slide, "LEAD主干与本项目扩展");
  const imageBytes = await fs.readFile(leadImagePath);
  slide.images.add({
    blob: imageBytes,
    contentType: "image/png",
    alt: "LEAD预训练、微调与被试级检测流程",
    fit: "contain",
    position: { left: 35, top: 110, width: 565, height: 300 },
  });
  rect(slide, 625, 118, 4, 280, BLUE);
  textbox(slide, "保留的LEAD能力", 655, 118, 250, 36, { fontSize: 21, bold: true, color: BLUE });
  bullets(slide, [
    "时间与空间联合编码",
    "异构通道和采样率适配",
    "被试级损失与预测聚合",
  ], 650, 165, 260, 125, { fontSize: 17, spaceAfterPoints: 5 });
  textbox(slide, "新增实验层", 655, 304, 250, 36, { fontSize: 21, bold: true, color: RED });
  bullets(slide, [
    "五频段输入与band embedding",
    "频段、电极和特征配对消融",
    "公开数据集上的跨折稳定性",
  ], 650, 350, 270, 125, { fontSize: 17, spaceAfterPoints: 5 });
  pageNo(slide, 9);
  note(slide, "图片来源：C:/Users/ROG/Desktop/LEAD/figs/lead_pipeline.png，来自LEAD项目。模型主体固定为LEAD。 ");
}

// 10. Experiments
{
  const slide = deck.slides.items[9];
  cleanBody(slide);
  setTitle(slide, "公开数据集是模型提升的主评测场");
  const chart = slide.charts.add("bar", {
    position: { left: 45, top: 115, width: 475, height: 300 },
    categories: ["APAVA（AD）", "RepOD（SZ）"],
    series: [
      { name: "片段级 AUROC", values: [0.9190, 0.6489], fill: BLUE },
      { name: "被试级 AUROC", values: [0.9000, 0.5667], fill: GOLD },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 90 },
    hasLegend: true,
    legend: { position: "bottom" },
    yAxis: { minimumScale: 0, maximumScale: 1, numberFormatCode: "0%", majorUnit: 0.2 },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0%" },
    chartFill: WHITE,
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: WHITE,
    plotAreaLine: { fill: GRID, width: 1 },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  textbox(slide, "当前进展", 565, 112, 320, 36, { fontSize: 21, bold: true, color: BLUE });
  bullets(slide, [
    "APAVA完成5次复现，被试级AUROC 90.00%",
    "RepOD完成5次MCCV，被试级AUROC 56.67%",
    "RepOD结果提示跨疾病迁移仍需优化",
  ], 560, 155, 350, 130, { fontSize: 17, spaceAfterPoints: 5 });
  textbox(slide, "后续比较", 565, 300, 320, 36, { fontSize: 21, bold: true, color: RED });
  bullets(slide, [
    "原始LEAD、随机初始化与传统基线",
    "单频段、删频段及MultiBand-LEAD",
    "同一被试折上比较AUROC与macro-F1",
  ], 560, 343, 350, 125, { fontSize: 17, spaceAfterPoints: 5 });
  textbox(slide, "两组数据疾病和规模不同，图中结果只表示阶段进展，不作数据集间优劣比较。", 55, 443, 850, 35, {
    fontSize: 15, color: MID, alignment: "center",
  });
  pageNo(slide, 10);
  note(slide, "数据来源：C:/Users/ROG/Desktop/LEAD/results/LEADv2/finetune/LEADv2/P-Base-F-APAVA-Multi/results.txt；C:/Users/ROG/Desktop/LEAD/results/LEADv2/finetune/LEADv2/P-Base-F-RepOD-MCCV/results.txt。APAVA：片段级AUROC 91.90%，被试级AUROC 90.00%。RepOD：片段级AUROC 64.89%，被试级AUROC 56.67%。 ");
}

toc(deck.slides.items[10], 3);

// 12. Roadmap
{
  const slide = deck.slides.items[11];
  cleanBody(slide);
  setTitle(slide, "后续实验路线与验证层级");
  const stages = [
    ["01", "公开数据基线", "固定被试划分\n复现LEAD与传统基线"],
    ["02", "因素筛选", "单频段与删频段\n电极和特征消融"],
    ["03", "模型提升", "频段门控融合\nMultiBand-LEAD"],
    ["04", "公开集验证", "跨数据集与多种子\n确认提升和排名稳定"],
    ["05", "医院外部验证", "模型定型后一次验证\n不参与调参"],
  ];
  stages.forEach(([num, head, body], i) => {
    const x = 37 + i * 181;
    textbox(slide, num, x, 130, 45, 34, { typeface: LATIN, fontSize: 18, bold: true, color: i === 4 ? RED : BLUE });
    rect(slide, x, 174, 145, 4, i === 4 ? RED : BLUE);
    textbox(slide, head, x, 193, 150, 44, { fontSize: 19, bold: true });
    textbox(slide, body, x, 246, 150, 86, { fontSize: 16, color: MID, verticalAlignment: "top" });
    if (i < 4) textbox(slide, "›", x + 147, 178, 28, 42, { typeface: LATIN, fontSize: 28, bold: true, color: GRID, alignment: "center" });
  });
  rect(slide, 48, 375, 850, 1, GRID);
  textbox(slide, "关键验收：相同公开数据、相同被试折和相同评价指标下，MultiBand-LEAD稳定优于原始LEAD。", 65, 392, 820, 44, {
    fontSize: 18, bold: true, color: RED, alignment: "center",
  });
  textbox(slide, "补充说明：资源允许时，仅在最终阶段用LaBraM复核关键频段与电极排名是否依赖LEAD架构。", 75, 447, 800, 35, {
    fontSize: 15, color: MID, alignment: "center",
  });
  pageNo(slide, 12);
  note(slide, "医院数据仅作为模型定型后的外部实验验证。LaBraM不进入主线开发，仅作为末期可选的模型独立性复核。 ");
}

// 13. Closing
{
  const slide = deck.slides.items[12];
  rect(slide, 0, 76, W, 428, WHITE);
  setTitle(slide, "总结");
  textbox(slide, "研究主线", 90, 130, 170, 40, { fontSize: 20, bold: true, color: BLUE });
  textbox(slide, "在公开数据集上提升LEAD，并定位稳定贡献分类的频段、电极和特征。", 245, 128, 620, 48, {
    fontSize: 22, bold: true,
  });
  textbox(slide, "验证边界", 90, 220, 170, 40, { fontSize: 20, bold: true, color: BLUE });
  textbox(slide, "医院数据用于最终外部验证，不驱动框架设计和模型选择。", 245, 218, 620, 48, {
    fontSize: 22, bold: true,
  });
  textbox(slide, "核心模型", 90, 310, 170, 40, { fontSize: 20, bold: true, color: BLUE });
  textbox(slide, "LEAD承担全部主实验；LaBraM仅作为末期补充对照。", 245, 308, 620, 48, {
    fontSize: 22, bold: true,
  });
  textbox(slide, "Thank you", 330, 414, 300, 44, { typeface: LATIN, fontSize: 28, bold: true, color: DARK, alignment: "center" });
  pageNo(slide, 13);
  note(slide, "汇报结束。 ");
}

const expectedSlideSizeEmu = "9144000,5143500";
const requirements = {
  explicitTotalSlideCount: 13,
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [10],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
};
const fontPolicy = {
  basis: "design",
  families: ["Arial", "宋体"],
  scriptFonts: { ea: "宋体" },
};

const stagingDir = path.join(workspaceDir, ".codex-finalizer-v2");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v2.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "6",
  ],
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [10],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "FYP版本2.pptx.validation.json"),
});

console.log(JSON.stringify(result, null, 2));
