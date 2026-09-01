import json, re, zipfile, xml.etree.ElementTree as ET

p='/Users/huaandre/Desktop/EEG文章筛选.xlsx'
ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with zipfile.ZipFile(p) as z:
    shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        root=ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root.findall('m:si',ns):
            shared.append(''.join(t.text or '' for t in si.iter('{%s}t'%ns['m'])))
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    relroot=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    rels={e.attrib['Id']:e.attrib['Target'] for e in relroot}
    sheets=[]
    for s in wb.find('m:sheets',ns):
        target=rels[s.attrib['{%s}id'%ns['r']]]
        if target.startswith('/'): target=target.lstrip('/')
        elif not target.startswith('xl/'): target='xl/'+target
        sheets.append((s.attrib['name'],target))
    out=[]
    for name,target in sheets:
        root=ET.fromstring(z.read(target))
        rows=[]; maxcol=0
        for row in root.findall('.//m:sheetData/m:row',ns):
            vals={}
            for c in row.findall('m:c',ns):
                ref=c.attrib['r']; letters=re.match(r'[A-Z]+',ref).group()
                col=0
                for ch in letters: col=col*26+ord(ch)-64
                typ=c.attrib.get('t'); v=c.find('m:v',ns); inline=c.find('m:is',ns)
                value=''
                if typ=='s' and v is not None: value=shared[int(v.text)]
                elif typ=='inlineStr' and inline is not None: value=''.join(t.text or '' for t in inline.iter('{%s}t'%ns['m']))
                elif v is not None: value=v.text
                vals[col]=value; maxcol=max(maxcol,col)
            rows.append(vals)
        matrix=[[r.get(i,'') for i in range(1,maxcol+1)] for r in rows]
        out.append({'sheet':name,'rows':len(matrix),'cols':maxcol,'data':matrix})
with open('/Users/huaandre/Desktop/FYP/.tmp_eeg_data.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps([{'sheet':x['sheet'],'rows':x['rows'],'cols':x['cols']} for x in out],ensure_ascii=False))
