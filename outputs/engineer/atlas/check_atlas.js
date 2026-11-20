/* Run: node outputs/engineer/atlas/check_atlas.js
   No packages, server, or browser required. Does not claim browser/layout coverage.
   Synthetic roles exist only in test memory, never in the delivered page or data. */
'use strict';
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=__dirname,html=fs.readFileSync(path.join(root,'atlas.html'),'utf8');
const scripts=Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g));
assert.strictEqual(scripts.length,1);
const code=scripts[0][1],context={};
vm.runInNewContext(code,context,{filename:'atlas.html'});
const C=context.AtlasCore;
const data=JSON.parse(fs.readFileSync(path.resolve(root,'../../../report/site/atlas.json'),'utf8'));
let checks=0;function check(name,fn){fn();checks++;console.log('PASS '+name);}
const S=(layer,kind,index,module)=>C.slot(layer,kind,index,module).key;
const head=(mod,ids,n)=>({param:mod,module:mod,layer:17,unit_kind:'head',units:ids.map(i=>[i,0.4]),n_units:n,n_units_changed:ids.length,n_values:100});
const mlp=(ids,changed)=>({param:'model.layers.8.mlp.down_proj.weight',module:'down_proj',layer:8,unit_kind:'neuron',units:ids.map(i=>[i,0.3]),n_units:8,n_units_changed:changed,n_values:100});
const edit=(id,t,kept)=>({id,step:id,kept_final:kept,accepted:kept,targets:[t]});
const a={run:'a',teacher:'A',task:'t1',student:'X',arch:{layers:26,heads:4,kv_heads:1},edits:[edit(1,head('k_proj',[0],1),true),edit(2,head('k_proj',[0],1),true),edit(3,mlp([1,2],8),true)]};
const b={run:'b',teacher:'B',task:'t1',student:'X',arch:a.arch,edits:[edit(1,head('v_proj',[0],1),true),edit(2,mlp([1,3],2),true)]};
const rejected={run:'c',teacher:'C',task:'t1',student:'X',arch:a.arch,edits:[edit(1,head('q_proj',[2],4),false)]};
check('KV is distinct from query-head numbering',()=>{assert.strictEqual(C.targetSlots(head('k_proj',[0],1))[0].kind,'kv');assert.strictEqual(C.targetSlots(head('o_proj',[0],4))[0].kind,'head');});
check('GQA links shared KV without inventing query-head edits',()=>{const idx=C.buildIndex([a,b],{},true);assert.strictEqual(idx.has(S(17,'head',0)),false);for(let h=0;h<4;h++)assert.strictEqual(C.merged(idx,C.gridKeys(17,h,a.arch)).editRuns.size,2);assert.strictEqual(idx.get(S(17,'kv',0)).editRuns.size,2);assert.strictEqual(idx.get(S(17,'kv',0)).editTeachers.size,2);});
check('Repeated edits within one run count once',()=>{const c=C.buildIndex([a],{},true).get(S(17,'kv',0));assert.strictEqual(c.edits.length,2);assert.strictEqual(c.editRuns.size,1);});
check('Final-only filter excludes rejected edits, all-attempts includes them',()=>{assert(!C.buildIndex([rejected],{},true).has(S(17,'head',2)));assert(C.buildIndex([rejected],{},false).has(S(17,'head',2)));});
check('Full-head changes survive top-24 truncation',()=>{const t=head('q_proj',Array.from({length:24},(_,i)=>i),32);t.n_units_changed=32;assert.strictEqual(C.targetSlots(t).filter(x=>x.kind==='head').length,32);});
check('Partially omitted heads remain explicitly unknown',()=>{const t=head('q_proj',[0],32);t.n_units_changed=3;const slots=C.targetSlots(t);assert.strictEqual(slots.filter(x=>x.kind==='head').length,1);assert.strictEqual(slots.filter(x=>x.kind==='other').length,1);});
check('Neuron overlap deduplicates runs, does not infer omitted IDs',()=>{a.edits.push(edit(4,mlp([1,2],8),true));const n=C.recurrentNeurons([a,b],true);assert.strictEqual(n.length,1);assert.strictEqual(n[0].index,1);assert.strictEqual(n[0].runs.size,2);});
check('MLP count is exact only when known; partial lists give bounds',()=>{let bounds=C.neuronBounds([{t:mlp([1,2],8)}]);assert.strictEqual(bounds.lower,8);assert.strictEqual(bounds.upper,8);bounds=C.neuronBounds([{t:mlp([1,2],4)},{t:mlp([2,3],4)}]);assert.strictEqual(bounds.lower,4);assert.strictEqual(bounds.upper,8);assert(!bounds.exact);});
check('Depth counts each edit-layer once, not each target or head',()=>{const r={...a,edits:[{...edit(1,head('k_proj',[0],1),true),targets:[head('k_proj',[0],1),head('v_proj',[0],1)]}]};const d=C.depthDistribution([r],true)[0];assert.strictEqual(d.total,1);assert.strictEqual(d.bins[6],1);});
check('Layer-only localization is never expanded to every head',()=>{const roles={a:{items:[{kind:'layer',layer:17,action:'probed'}]}};const idx=C.buildIndex([{...a,edits:[]}],roles,true);assert.strictEqual(idx.size,1);assert.strictEqual(idx.get(S(17,'layer',null)).roleRuns.size,1);});
check('Norm role and actual tensor edit share a component',()=>{const t={layer:17,module:'q_norm',unit_kind:'tensor'};const role={layer:17,module:'q_norm',kind:'norm'};assert.strictEqual(C.targetSlots(t)[0].key,C.roleSlots(role)[0].key);});
check('Probe history survives final-only filter, rejected edit explanation does not',()=>{const roles={c:{items:[{action:'edited',edit_ids:[1]},{action:'ablated',edit_ids:[]}]}};assert.strictEqual(C.roleItems(roles,rejected,true).length,1);assert.strictEqual(C.roleItems(roles,rejected,false).length,2);});
const counts=[];
for(const pair of [['t2','google/gemma-3-1b-it'],['t1','Qwen/Qwen3-1.7B']]){
  const runs=data.runs.filter(r=>r.task===pair[0]&&r.student===pair[1]);
  check('Real data group exists: '+pair.join(' / '),()=>assert(runs.length>0));
  for(const only of [true,false]){
    const idx=C.buildIndex(runs,{},only),neurons=C.recurrentNeurons(runs,only),depth=C.depthDistribution(runs,only);
    check('All real group counters respect run/teacher denominators: '+pair.join('/')+' '+only,()=>{for(const c of idx.values()){assert(c.editRuns.size<=runs.length);assert(c.editTeachers.size<=new Set(runs.map(r=>r.teacher)).size);}assert(neurons.every(n=>n.runs.size>=2&&n.runs.size<=runs.length));assert(depth.every(d=>d.bins.reduce((s,n)=>s+n,0)===d.total));});
    counts.push({task:pair[0],student:pair[1],final_only:only,runs:runs.length,edits:runs.reduce((n,r)=>n+C.editsFor(r,only).length,0),components:idx.size,recurrent_top_neurons:neurons.length});
  }
}
// Minimal DOM interface exercises the actual page initialization/render/event code.
// It intentionally does not pretend to validate browser CSS, layout, or accessibility.
class Element{
  constructor(id){this.id=id;this.value=id==='theme'?'auto':id==='sort'?'edits':'';this.checked=id==='kept';this.hidden=false;this.disabled=false;this.innerHTML='';this.textContent='';this.dataset={};this.events={};}
  addEventListener(name,fn){this.events[name]=fn;}setAttribute(){}scrollIntoView(){}
}
async function app(roles,options={}){
  const elements=new Map(),listeners={},doc={documentElement:{dataset:{}},getElementById(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);},querySelectorAll(){return [];},addEventListener(name,fn){listeners[name]=fn;}};
  const errors=[],appContext={document:doc,console:{log(){},warn(){},error(...x){errors.push(x);}},localStorage:{getItem(){return null;},setItem(){}},setTimeout,clearTimeout,AbortController:class{constructor(){this.signal={};}abort(){}},
    fetch:async url=>{if(url==='site/atlas.json'){if(options.dataError)throw Error('network unavailable');return {ok:true,json:async()=>options.empty?{runs:[]}:data};}if(options.rolesError)throw Error('roles missing');return {ok:true,json:async()=>roles};}};
  vm.runInNewContext(code,appContext,{filename:'atlas.html'});
  await new Promise(resolve=>setImmediate(resolve));
  return {elements,doc,errors,listeners,el:id=>doc.getElementById(id)};
}
(async()=>{
  const t2=data.runs.find(r=>r.task==='t2'&&r.student==='google/gemma-3-1b-it');
  const t1=data.runs.find(r=>r.task==='t1'&&r.student==='Qwen/Qwen3-1.7B');
  const fixtures={
    [t2.run]:{summary_zh:'测试摘要：头部定位',items:[{label:'L17H0',layer:17,kind:'head',module:'o_proj',index:0,role_zh:'测试解释 <img src=x onerror=alert(1)>',evidence_zh:'仅供测试的定位依据',action:'probed',edit_ids:[],step:2,source:'test-only'}]},
    [t1.run]:{summary_zh:'测试摘要：MLP 编辑',items:[{label:'L8MLP',layer:8,kind:'mlp',module:'down_proj',index:null,role_zh:'测试解释：规则提取',evidence_zh:'仅供测试的编辑依据',action:'edited',edit_ids:[t1.edits[0].id],step:t1.edits[0].step,source:'test-only'}]}
  };
  const ui=await app(fixtures);
  check('Page initialization renders all four sections with real data',()=>{assert.strictEqual(ui.doc.documentElement.dataset.ready,'true');for(const id of ['grid','events','common-table','depth-chart'])assert(ui.el(id).innerHTML.length>100);assert.strictEqual(ui.errors.length,0);});
  check('Two synthetic roles render and escape untrusted text',()=>{ui.el('run').value=t2.run;ui.el('run').events.change();assert(ui.el('events').innerHTML.includes('&lt;img'));assert(!ui.el('events').innerHTML.includes('<img'));assert(ui.el('run-intro').innerHTML.includes('测试摘要：头部定位'));});
  check('Switch to MedCalc + Qwen 1.7B renders linked role and timeline',()=>{ui.el('task').value='t1';ui.el('task').events.change();ui.el('student').value='Qwen/Qwen3-1.7B';ui.el('student').events.change();ui.el('run').value=t1.run;ui.el('run').events.change();assert(ui.el('run-intro').innerHTML.includes('测试摘要：MLP 编辑'));assert(ui.el('events').innerHTML.includes('测试解释：规则提取'));assert(ui.el('grid').innerHTML.includes('H15'));});
  check('All-attempts toggle and sorting rerender without runtime errors',()=>{ui.el('kept').checked=false;ui.el('kept').events.change();assert(ui.el('scope-note').textContent.includes('全部改动尝试'));ui.el('sort').value='roles';ui.el('sort').events.change();assert(ui.el('common-table').innerHTML.includes('定位教师'));});
  check('Cell details are reachable through delegated clicks',()=>{ui.listeners.click({target:{closest(sel){return sel==='[data-cell]'?{dataset:{cell:'0'}}:null;}}});assert(ui.el('detail').innerHTML.includes('部位')||ui.el('detail').innerHTML.includes('L0'));});
  const optional=await app({}, {rolesError:true});
  check('Missing roles preserve working real-data page with visible notice',()=>{assert.strictEqual(optional.doc.documentElement.dataset.ready,'true');assert(optional.el('roles-status').textContent.includes('暂不可用'));assert(optional.el('events').innerHTML.length>100);});
  const missing=await app({}, {dataError:true});
  check('Unavailable primary data shows actionable error and retry',()=>{assert.strictEqual(missing.el('content').hidden,true);assert(missing.el('load-status').textContent.includes('无法读取'));assert.strictEqual(missing.el('reload').disabled,false);});
  const empty=await app({}, {empty:true});
  check('Empty dataset renders graceful state',()=>assert(empty.el('load-status').textContent.includes('没有运行')));
  const rolePath=path.resolve(root,'../../../report/site/atlas_roles.json');
  const rolesExist=fs.existsSync(rolePath);
  if(rolesExist){
    const realRoles=JSON.parse(fs.readFileSync(rolePath,'utf8')),realUI=await app(realRoles);
    check('Real roles load without fallback warning',()=>{assert.strictEqual(realUI.doc.documentElement.dataset.ready,'true');assert.strictEqual(realUI.el('roles-status').hidden,true);});
    check('All real runs render actual summaries, timeline, and role details',()=>{
      for(const r of data.runs){
        realUI.el('task').value=r.task;realUI.el('task').events.change();
        realUI.el('student').value=r.student;realUI.el('student').events.change();
        realUI.el('run').value=r.run;realUI.el('run').events.change();
        assert(realUI.el('run-intro').innerHTML.includes(r.run));
        assert(realUI.el('events').innerHTML.length>0);
        assert(realRoles[r.run]&&realRoles[r.run].summary_zh);
        for(const only of [true,false]){
          realUI.el('kept').checked=only;realUI.el('kept').events.change();
          assert(!realUI.el('events').innerHTML.includes('NaN'));
          const idx=C.buildIndex([r],realRoles,only);
          for(const c of idx.values())assert(c.roleRuns.size<=1);
          const first=C.roleItems(realRoles,r,only)[0];
          if(first){const k=C.roleSlots(first)[0].key;realUI.listeners.click({target:{closest(sel){return sel==='[data-part]'?{dataset:{part:k}}:null;}}});assert(realUI.el('detail').innerHTML.includes('role'));}
        }
      }
      assert.strictEqual(realUI.errors.length,0);
    });
    check('Real multi-head roles retain each explicit query-head index',()=>{
      const role=Object.values(realRoles).flatMap(x=>x.items).find(i=>i.kind==='head'&&Array.isArray(i.index)&&i.index.length>1);
      assert(role);const slots=C.roleSlots(role);assert.strictEqual(slots.length,role.index.length);assert(slots.every(s=>role.index.includes(s.index)));
    });
  }
  const report={checks_passed:checks,coverage:'Pure aggregation + actual render/event code under minimal DOM; not a browser/layout test.',real_data_counts:counts,real_roles_tested:rolesExist,fixture_roles_written_to_app:false};
  fs.writeFileSync(path.join(root,'checks.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
