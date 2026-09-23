/* Mosquito scanner, AGPL-3.0-or-later. Source: source.zip.
   Wingbeat AI: feeds the microphone to the HumBug MozzBNN detector (sound-worker.js) every 0.5 s over the last 1.92 s.
   A window counts as a hit only when the model says mosquito (MC-dropout mean > 0.5) AND the 250–1000 Hz band has a
   clear wingbeat peak (>= 9.5 dB above the band median); this rejects broadband noise (fans, air conditioning) that the
   per-window normalised model alone can mistake for mosquitoes. Two hits in the last three analyses mark a detection. */
(()=>{
const q=id=>document.getElementById(id),box=q('soundAi');
const P_MIN=.5,PEAK_DB=9.5,HISTORY=3,NEED=2,STEP_MS=500,SPAN_S=2.1;
let worker=null,rpcId=0,pending=new Map(),gen=0,node=null,sink=null,timer=0,ring=null,filled=0,pos=0,rate=0,busy=false,history=[];
const show=(s,cls='')=>{box.textContent=s;box.className='soundAi'+(cls?' '+cls:'')};
function rpc(msg,transfer=[]){if(!worker){worker=new Worker('sound-worker.js');worker.onmessage=e=>{const m=e.data;if(m.progress){show(m.progress);return}const p=pending.get(m.id);if(!p)return;pending.delete(m.id);m.error?p.reject(Error(m.error)):p.resolve(m)};worker.onerror=()=>{for(const p of pending.values())p.reject(Error('소리 AI 실행에 실패했습니다. 새로고침 후 다시 시도하세요.'));pending.clear();worker.terminate();worker=null}}return new Promise((resolve,reject)=>{const id=++rpcId;pending.set(id,{resolve,reject});worker.postMessage({...msg,id},transfer)})}
function teardown(){clearInterval(timer);timer=0;try{node?.port.close();node?.disconnect();sink?.disconnect()}catch{}node=sink=null;ring=null;busy=false;history=[];aiSoundOwnsTitle=false}
addEventListener('mosquito-mic-stop',()=>{gen++;teardown();show('마이크를 켜면 모기 날갯소리 AI(HumBug)가 함께 분석합니다.')});
addEventListener('mosquito-mic',async e=>{const {ac,source}=e.detail,g=++gen;teardown();if(!window.AudioWorkletNode||!ac.audioWorklet){show('이 브라우저는 소리 AI(AudioWorklet)를 지원하지 않습니다. 주파수 그래프만 표시합니다.');return}show('소리 AI 준비 중');try{await Promise.all([rpc({type:'init'}),ac.audioWorklet.addModule('audio-tap.js')]);if(g!==gen)return;rate=ac.sampleRate;ring=new Float32Array(Math.ceil(rate*SPAN_S));filled=pos=0;node=new AudioWorkletNode(ac,'mosquito-tap',{numberOfOutputs:1});sink=ac.createGain();sink.gain.value=0;source.connect(node);node.connect(sink).connect(ac.destination);node.port.onmessage=m=>{const b=m.data;for(let i=0;i<b.length;i++){ring[pos]=b[i];pos=(pos+1)%ring.length}filled=Math.min(ring.length,filled+b.length)};aiSoundOwnsTitle=true;show('소리 AI 듣는 중 · 휴대폰 마이크를 모기 가까이(10–20 cm) 두세요');timer=setInterval(()=>analyse(g),STEP_MS)}catch(err){if(g!==gen)return;teardown();show('소리 AI 실행 실패: '+err.message+' · 주파수 그래프만 표시합니다.')}});
async function analyse(g){if(busy||!ring||filled<ring.length||document.hidden)return;busy=true;const x=new Float32Array(ring.length);x.set(ring.subarray(pos));x.set(ring.subarray(0,pos),ring.length-pos);try{const r=await rpc({type:'infer',samples:x,rate},[x.buffer]);if(g!==gen)return;const hit=r.p>P_MIN&&r.prominence>=PEAK_DB;history=history.concat(hit).slice(-HISTORY);const detected=history.filter(Boolean).length>=NEED,certain=r.pe<.5?'불확실성 낮음':'불확실성 높음';
 q('soundTitle').textContent=detected?'모기 날갯소리 추정 · AI':audioHeld>15?'유사 대역 소리 · AI는 모기 소리로 보지 않음':'주변 소리 분석 중';
 if(detected)show(`모기 날갯소리로 추정 · AI ${r.p.toFixed(2)} · 봉우리 ${Math.round(r.peakHz)} Hz · ${certain} (확정 판정 아님)`,'hit');
 else if(hit)show(`모기 소리 후보 · AI ${r.p.toFixed(2)} · 봉우리 ${Math.round(r.peakHz)} Hz · 연속 확인 중`);
 else if(r.p>P_MIN)show(`AI가 반응했지만 날갯짓 주파수 봉우리가 약합니다 (${r.prominence.toFixed(1)} dB) · 선풍기·에어컨 등 잡음일 수 있음`);
 else show(`소리 AI 듣는 중 · 모기 소리 아님 (AI ${r.p.toFixed(2)})`)}catch(err){if(g!==gen)return;teardown();show('소리 AI 오류: '+err.message)}finally{if(g===gen)busy=false}}
})();
