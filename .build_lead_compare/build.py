from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from html import escape
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'LEAD和LaBraM对比.pptx'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
P,A,R=NS['p'],NS['a'],NS['r']
def el(tag,**attrs): return E.Element(tag,**{k:str(v) for k,v in attrs.items()})
def sub(parent,tag,**attrs):
    x=el(tag,**attrs); parent.append(x); return x
def q(n): return '{'+NS[n[0]]+'}'+n[2:]
def fill(parent,color): sub(sub(parent,q('a:solidFill')),q('a:srgbClr'),val=color)
def textbody(parent,text,size=20,color='243449',bold=False):
    tx=sub(parent,q('p:txBody')); sub(tx,q('a:bodyPr'),wrap='square',lIns=0,rIns=0,tIns=0,bIns=0); sub(tx,q('a:lstStyle'))
    for line in text.split('\n'):
        p=sub(tx,q('a:p')); pp=sub(p,q('a:pPr')); sub(pp,q('a:buNone')); r=sub(p,q('a:r')); rp=sub(r,q('a:rPr'),lang='zh-CN',sz=int(size*100),b=int(bold)); fill(rp,color)
        sub(rp,q('a:latin'),typeface='Microsoft YaHei'); sub(rp,q('a:ea'),typeface='Microsoft YaHei'); sub(r,q('a:t')).text=line
    return tx
def shape(tree,id,text,x,y,w,h,size=20,color='243449',bold=False):
    s=sub(tree,q('p:sp')); nv=sub(s,q('p:nvSpPr')); sub(nv,q('p:cNvPr'),id=id,name='Text '+str(id)); sub(nv,q('p:cNvSpPr'),txBox=1); sub(nv,q('p:nvPr'))
    sp=sub(s,q('p:spPr')); xf=sub(sp,q('a:xfrm')); sub(xf,q('a:off'),x=int(x*12700),y=int(y*12700)); sub(xf,q('a:ext'),cx=int(w*12700),cy=int(h*12700)); sub(sub(sp,q('a:prstGeom'),prst='rect'),q('a:avLst')); sub(sp,q('a:noFill')); sub(sub(sp,q('a:ln')),q('a:noFill')); textbody(s,text,size,color,bold)
def table(tree,id,rows,widths,y=132,rh=58,size=18):
    gf=sub(tree,q('p:graphicFrame')); nv=sub(gf,q('p:nvGraphicFramePr')); sub(nv,q('p:cNvPr'),id=id,name='Comparison table'); sub(nv,q('p:cNvGraphicFramePr')); sub(nv,q('p:nvPr')); xf=sub(gf,q('p:xfrm')); sub(xf,q('a:off'),x=44*12700,y=y*12700); sub(xf,q('a:ext'),cx=sum(widths)*12700,cy=len(rows)*rh*12700)
    gd=sub(sub(gf,q('a:graphic')),q('a:graphicData'),uri=A+'/table'); tbl=sub(gd,q('a:tbl')); sub(tbl,q('a:tblPr'),firstRow=1,bandRow=0); grid=sub(tbl,q('a:tblGrid'))
    for w in widths: sub(grid,q('a:gridCol'),w=w*12700)
    for i,row in enumerate(rows):
        tr=sub(tbl,q('a:tr'),h=rh*12700)
        for j,v in enumerate(row):
            tc=sub(tr,q('a:tc')); tx=textbody(tc,v,size,'FFFFFF' if i==0 else '243449',i==0); tx.tag=q('a:txBody'); prop=sub(tc,q('a:tcPr'),marL=12*12700,marR=9*12700,marT=4*12700,marB=3*12700,anchor='ctr'); fill(prop,'16324F' if i==0 else ('EEF3F7' if i%2 else 'FFFFFF'))

slides=[]
def add(title,lines=None,rows=None,widths=None,foot='',note='',cover=False): slides.append(dict(title=title,lines=lines,rows=rows,widths=widths,foot=foot,note=note,cover=cover))
add('LEAD 和 LaBraM 对比', ['EEG 预训练模型与公开数据集实验','LEAD 采用项目中的 LEADv2 P-Base','FYP 研究汇报    2026.10'],cover=True)
add('本次对比的主要发现', ['AD：LEADv2 的片段指标较高，患者级均为 6/7。','FTD：患者级均为 6/6，片段比较仍需核实配置。','SZ：LaBraM 五次划分的平均指标较高，但波动明显。','MDD：LEADv2 指标较高，多数投票 100% 对 83.33%。'],foot='结果来自当前微调实现，不支持“某模型在所有疾病上更好”的结论。')
add('模型表示与项目角色',rows=[['比较维度','LEADv2','LaBraM'],['表示方式','单通道 EEG patch\n时间、空间及采样率信息','EEG 通道 patch\n离散神经编码'],['建模 / 预训练','时间与通道维度\n注意力建模','预测遮蔽 patch 的\n离散神经编码'],['当前权重','P-Base 预训练权重','base 预训练权重'],['项目角色','主模型，电极×频段研究','独立模型对照']],widths=[140,246,246],foot='LaBraM 通过适配器读取 LEAD 格式数据，未复现原论文完整评测协议。')
add('任务与评估口径',rows=[['公开数据集','分类任务','本次测试方案'],['ADFTD-RS','AD vs CN','固定被试划分，7 人'],['ADFTD-RS','FTD vs CN','固定被试划分，6 人'],['Mumtaz','MDD vs HC','固定划分，LaBraM 测试 6 人'],['RepOD','SZ vs HC','seed 41–45，每次 6 人']],widths=[140,250,242],foot='主指标：片段准确率、片段 AUROC、患者级多数投票准确率。',note='RepOD 为五次重复随机划分 MCCV，测试被试可能重复，并非标准五折交叉验证。')
add('AD：片段优势未转化为患者级差距',rows=[['指标','LEADv2','LaBraM'],['片段准确率','79.34%','73.25%'],['片段 AUROC','89.59%','81.89%'],['患者级准确率','85.71%（6/7）','85.71%（6/7）']],widths=[250,191,191],foot='200 Hz 完整运行。LEADv2 准确率高 6.09 个百分点，AUROC 高 7.70 个百分点。',note='ADFTD-RS AD vs CN。LaBraM 测试片段 5,562，AUROC 使用独立评估脚本结果，不使用训练日志口径。')
add('FTD：患者级持平，配置仍待补证',rows=[['指标','LEADv2','LaBraM'],['片段准确率','92.74%','83.80%'],['片段 AUROC','98.01%','94.98%'],['患者级准确率','100%','100%（6/6）']],widths=[250,191,191],foot='LEAD 旧结果未保存实际采样率和测试片段数，片段差距暂不作为严格同条件结论。',note='LaBraM 为 200 Hz、seed 41，4,747 测试片段。其一名 FTD 平均预测概率约 0.543。训练日志曾出现 Infinity / NaN 梯度范数，需重复实验检查。')
add('MDD：LEADv2 的当前指标较高',rows=[['指标','LEADv2','LaBraM'],['片段准确率','89.91%','83.14%'],['片段 AUROC','98.78%','93.57%'],['患者级多数投票','100.00%','83.33%（5/6）']],widths=[250,191,191],foot='Mumtaz 固定划分，200 Hz 标记运行，训练 seed 41。LEAD 测试 ID 尚待补齐。',note='LaBraM 测试 6 人、1,714 片段。平均概率投票为 100%（6/6），本表统一采用多数投票。被试 123 平均概率 0.50763，但多数投票判错。LEAD 结果未保存测试 ID、片段数及逐人预测，不能直接声称已验证同一组 6/6。固定划分加载器使用 random.Random(42)。来源：C:/Users/ROG/Desktop/LEAD/results/LEADv2/finetune/LEADv2/P-Base-F-MDD-Mumtaz-200Hz-Fixed/results.txt；C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_seed41/subject_metrics.json；C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_seed41/subject_predictions.csv')
add('SZ：LaBraM 的五次平均表现更高',rows=[['指标','LEADv2','LaBraM'],['片段准确率','61.79 ± 6.73%','70.93 ± 9.33%'],['片段 AUROC','62.64 ± 10.37%','79.45 ± 3.86%'],['患者级准确率','56.67 ± 8.16%','76.67 ± 17.00%']],widths=[236,198,198],foot='200 Hz，seed 41–45。“±”表示五次结果的总体标准差，不是置信区间。',note='LaBraM 平均片段准确率高 9.14 个百分点，片段 AUROC 高 16.81 个百分点，患者级准确率高 20.00 个百分点。seed43 使用成功重跑 seed43_new。')
add('SZ：患者级结果对划分较敏感',rows=[['模型 / seed','41','42','43','44','45'],['LEADv2','4/6','3/6','3/6','4/6','3/6'],['LaBraM','5/6','4/6','3/6','5/6','6/6']],widths=[192,88,88,88,88,88],foot='每次仅 6 名测试被试，一人判断变化对应约 16.67 个百分点。',note='LaBraM 患者级准确率范围 50%–100%，LEADv2 范围 50%–66.67%。不能将五次的测试人数简单视为 30 名独立被试。')
add('四任务结果总表',rows=[['任务','片段准确率','片段 AUROC','患者级准确率'],['AD','79.34 / 73.25','89.59 / 81.89','85.71 / 85.71'],['FTD*','92.74 / 83.80','98.01 / 94.98','100 / 100'],['MDD','89.91 / 83.14','98.78 / 93.57','100 / 83.33'],['SZ 均值','61.79 / 70.93','62.64 / 79.45','56.67 / 76.67']],widths=[116,172,172,172],foot='每格依次为 LEADv2 / LaBraM，单位 %。患者级采用多数投票。FTD* 配置待补证。',note='SZ 为五次 MCCV 均值，标准差见 SZ 结果页。其他任务为单次固定划分。MDD 的 LEAD 实际测试 ID 尚待补齐。')
add('比较结果的适用范围', ['模型设置：训练策略、patch 和预训练来源不同。','FTD 配置：需补齐 LEAD 采样率与实际测试片段数。','指标口径：患者级多数投票与平均概率不能混用。','小样本：单次 100% 不能代表可靠临床性能。'],foot='FTD 运行记录参数量约为 LEADv2 332 万、LaBraM 582 万，属于当前配置。')
add('电极与频段稳定性研究', ['LEADv2 主干：全频段、逐频段及逐电极遮蔽基线。','组合遮蔽：比较电极×频段组合对患者级指标的影响。','稳定性：记录不同被试划分中重要特征的重合度。','LaBraM 复核：检验关键筛选结论是否依赖单一模型。'],foot='分类贡献只表示关联，不能解释为疾病的因果因素。')
add('后续实验安排', ['首先补证 FTD 配置，必要时重跑明确标记的 200 Hz 基线。','为 AD、FTD 与 MDD 固定多次被试划分，并保存实际被试 ID。','在训练集与内层验证集完成掩码、超参数及搜索选择。','方法固定后，用医院数据开展外部验证。'],foot='已有实验查看过测试指标，后续比较应预先固定评估方案。',note='掩码基线建立后，再探索受 EvoFuse 启发的掩码或任务权重搜索，与随机搜索、贪心筛选及无掩码 LEADv2 比较。控制跨数据集设备、参考及处理流程差异。')
add('资料来源与结论', ['项目资料：LEADv2 与 LaBraM 公开数据集对比报告','报告更新：2026-10-10','当前结果支持继续做电极与频段稳定性研究。','LEADv2 作为主模型，LaBraM 用于跨模型复核。'],foot='原始运行记录路径及各任务指标口径见相应页面备注。')

source=(ROOT/'LEADv2与LaBraM公开数据集对比报告.md').read_text(encoding='utf-8')
with ZipFile(ROOT/'outputs'/'FYP版本2.pptx') as z: files={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
for n in list(files):
    if re.match(r'ppt/(slides|notesSlides|charts|media)/',n): del files[n]
pr=E.fromstring(files['ppt/presentation.xml']); ids=pr.find(q('p:sldIdLst')); ids.clear()
rels=E.fromstring(files['ppt/_rels/presentation.xml.rels']); RN='{http://schemas.openxmlformats.org/package/2006/relationships}'
for e in list(rels):
    if e.get('Type','').endswith('/slide'): rels.remove(e)
ct=E.fromstring(files['[Content_Types].xml']); CN='{http://schemas.openxmlformats.org/package/2006/content-types}'
for e in list(ct):
    if re.match(r'/ppt/(slides|notesSlides|charts|media)/',e.get('PartName','')): ct.remove(e)
for i,d in enumerate(slides,1):
    root=el(q('p:sld'),nsmap=NS) if False else E.Element(q('p:sld'),nsmap=NS)
    cs=sub(root,q('p:cSld')); bg=sub(cs,q('p:bg')); bp=sub(bg,q('p:bgPr')); fill(bp,'FFFFFF'); sub(bp,q('a:effectLst')); tree=sub(cs,q('p:spTree')); nv=sub(tree,q('p:nvGrpSpPr')); sub(nv,q('p:cNvPr'),id=1,name=''); sub(nv,q('p:cNvGrpSpPr')); sub(nv,q('p:nvPr')); gp=sub(tree,q('p:grpSpPr')); xf=sub(gp,q('a:xfrm'))
    for tag,at in [('off',{'x':0,'y':0}),('ext',{'cx':0,'cy':0}),('chOff',{'x':0,'y':0}),('chExt',{'cx':0,'cy':0})]: sub(xf,q('a:'+tag),**at)
    shape(tree,2,d['title'],44,90 if d['cover'] else 36,632,78 if d['cover'] else 66,40 if d['cover'] else 30,'16324F',True)
    if d['rows']: table(tree,3,d['rows'],d['widths'],102 if len(d['rows'])==5 else 110,46 if len(d['rows'])==5 else 58,17 if len(d['rows'])==5 else 18)
    if d['lines']:
        for j,line in enumerate(d['lines']): shape(tree,4+j,line,44,(205 if d['cover'] else 126)+j*(48 if d['cover'] else 56),632,44,20 if d['cover'] else 21,'243449',False)
    if d['foot']: shape(tree,20,d['foot'],44,354,632,43,14,'526276')
    shape(tree,21,f'{i:02d}',660,382,30,18,10,'6C7988')
    sub(sub(root,q('p:clrMapOvr')),q('a:masterClrMapping'))
    files[f'ppt/slides/slide{i}.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    sr=el(RN+'Relationships'); sub(sr,RN+'Relationship',Id='rId1',Type=R+'/slideLayout',Target='../slideLayouts/slideLayout1.xml'); sub(sr,RN+'Relationship',Id='rId2',Type=R+'/notesSlide',Target=f'../notesSlides/notesSlide{i}.xml'); files[f'ppt/slides/_rels/slide{i}.xml.rels']=E.tostring(sr)
    notes=E.Element(q('p:notes'),nsmap=NS); nc=sub(notes,q('p:cSld')); nt=sub(nc,q('p:spTree')); nn=sub(nt,q('p:nvGrpSpPr')); sub(nn,q('p:cNvPr'),id=1,name=''); sub(nn,q('p:cNvGrpSpPr')); sub(nn,q('p:nvPr')); sub(nt,q('p:grpSpPr'))
    shape(nt,2,d['note']+'\n来源：'+str(ROOT/'LEADv2与LaBraM公开数据集对比报告.md')+'\n\n原始报告与运行路径：\n'+source,20,20,500,650,12)
    nsp=nt.find(q('p:sp')+'/'.join(['',q('p:nvSpPr'),q('p:nvPr')]))
    if nsp is not None: sub(nsp,q('p:ph'),type='body',idx=1)
    sub(sub(notes,q('p:clrMapOvr')),q('a:masterClrMapping')); files[f'ppt/notesSlides/notesSlide{i}.xml']=E.tostring(notes)
    nr=el(RN+'Relationships'); sub(nr,RN+'Relationship',Id='rId1',Type=R+'/slide',Target=f'../slides/slide{i}.xml'); sub(nr,RN+'Relationship',Id='rId2',Type=R+'/notesMaster',Target='../notesMasters/notesMaster1.xml'); files[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=E.tostring(nr)
    sub(ids,q('p:sldId'),id=255+i,**{q('r:id'):f'rId{100+i}'}); sub(rels,RN+'Relationship',Id=f'rId{100+i}',Type=R+'/slide',Target=f'slides/slide{i}.xml')
    for part,typ in [(f'/ppt/slides/slide{i}.xml','slide'),(f'/ppt/notesSlides/notesSlide{i}.xml','notesSlide')]: sub(ct,CN+'Override',PartName=part,ContentType=f'application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml')
files['ppt/presentation.xml']=E.tostring(pr); files['ppt/_rels/presentation.xml.rels']=E.tostring(rels); files['[Content_Types].xml']=E.tostring(ct)
OUT.parent.mkdir(exist_ok=True)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for n,b in files.items(): z.writestr(n,b)
print(OUT)
print('Slides:',len(slides),'Native tables:',sum(bool(d['rows']) for d in slides))


