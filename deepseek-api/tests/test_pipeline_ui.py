import json
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / 'static' / 'pipeline-controls.js'

class PipelineUITests(unittest.TestCase):
    def test_interpretation_progress_and_resolved_query_are_visible_as_text(self):
        self.run_js(r'''
const assert=require('node:assert/strict');
class El {constructor(){this.children=[];this.dataset={};this.textContent='';} append(...x){this.children.push(...x);} setAttribute(){} }
global.document={createElement:()=>new El()};
const card={run_id:'r',query:'Найди Python',steps:[],interpretation:{status:'running'}};
assert.ok(p.render(card).children.some(e=>e.textContent.includes('Понимаю запрос')));
card.interpretation={status:'success',query:'<tag> language:python',limit:2};
assert.ok(p.render(card).children.some(e=>e.textContent.includes('<tag> language:python · до 2')));
''')

    def run_js(self, program):
        self.assertTrue(SCRIPT.exists(), 'pipeline UI is missing')
        result = subprocess.run(['node', '-e', f'const p=require({json.dumps(str(SCRIPT))});\n'+program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unicode_chunking_progress_before_terminal_and_duplicate_sequence(self):
        self.run_js(r'''
const assert=require('node:assert/strict');
(async()=>{
 let controller;
 const response=new Response(new ReadableStream({start(c){controller=c;}}));
 const seen=[]; const promise=p.consume(response,e=>seen.push(e));
 const enc=new TextEncoder();
 function push(e){const b=enc.encode(JSON.stringify(e)+'\n'); for(let i=0;i<b.length;i++)controller.enqueue(b.slice(i,i+1));}
 const base={run_id:'r',agent_id:'agent-1',branch_id:'main'};
 push({...base,seq:1,type:'pipeline_started',query:'Привет'});
 push({...base,seq:2,type:'step_completed',step:'search'});
 push({...base,seq:2,type:'step_completed',step:'search'});
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(seen.length,2); assert.equal(seen[1].step,'search');
 push({...base,seq:3,type:'pipeline_completed',agent:{id:'agent-1'},status:'success'});
 controller.close();
 const final=await promise;assert.equal(final.type,'pipeline_completed');assert.equal(seen.length,3);
})().catch(e=>{console.error(e);process.exit(1);});
''')

    def test_eof_without_terminal_is_not_success(self):
        self.run_js(r'''
const assert=require('node:assert/strict');
(async()=>{await assert.rejects(p.consume(new Response('{"run_id":"r","agent_id":"a","branch_id":"b","seq":1,"type":"pipeline_started"}\n'),()=>{}),/прервана/);})().catch(e=>{console.error(e);process.exit(1);});
''')

    def test_renderer_shows_actual_counts_and_json_as_text(self):
        self.run_js(r'''
const assert=require('node:assert/strict');
class El {constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.textContent='';} append(...x){this.children.push(...x);} setAttribute(){} addEventListener(){} }
global.document={createElement:tag=>new El(tag)};
const root=p.render({run_id:'r',query:'mcp',status:'running',steps:[{name:'search',status:'success',arguments:{query:'mcp'},result:{structuredContent:{items:[{name:'<script>'}]}},durationMs:2,serverUrl:'local'},{name:'summarize',status:'running',arguments:{}}]});
const all=[];function visit(e){all.push(e);e.children.forEach(visit);}visit(root);
assert.ok(all.some(e=>e.textContent.includes('найдено 1')));
assert.ok(all.some(e=>e.textContent.includes('готовлю обзор')));
assert.ok(all.some(e=>e.tag==='pre' && e.textContent.includes('<script>')));
assert.ok(all.some(e=>e.tag==='details'));
''')
