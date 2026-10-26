"""Reproducible WP-architecture Gaussian denoising study, exact compact SVD.

Run with C:/anaconda3/python.exe -s runner.py <benchmark|test|search|main|evaluate|all>.
The runtime module loads installed NumPy/SciPy before adding the PyTorch user site.
"""
import runtime
from runtime import np, scipy, torch
import argparse, copy, csv, hashlib, json, math, os, platform, time, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from model import Denoiser
from storage import write_json, read_json, write_artifact, resolve_artifact, artifact_exists, list_artifacts

SOURCE_ROOT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('MUON_NONLINEAR_OUTPUT',str(SOURCE_ROOT))).resolve()
FAMILIES=['lognormal','powerlaw','geometric','two_band']
METHODS=['sgd','muon']
DEFAULT={
 'dimension':128,'seeds':[17,43,97],'search_seed':137,'batch_size':1024,
 'search_steps':512,'main_steps':5000,'validation_n':4096,'test_n':8192,
 'fit_n_each':8192,'fixed_test_n':8192,'fixed_sigmas':[0.1,1.0,3.0],
 'sgd_lrs':[0.001,0.01,0.1,1.0],'muon_lrs':[0.0001,0.001,0.01,0.1],
 'boundary_extension_factor':10,'boundary_max_rounds':2,
 'aux_lr':'family_selected_sgd_lr','schedule':'constant','polar_workers':2,
 'recovery_interval':100,'search_budget':'same base four and maximum eight candidates per method; adaptive counts reported',
 'checkpoint_steps':[0,25,50,100,200,400,800,1200,2000,3000,4000,5000],
 'checkpoint_interval':100,'matched_loss_targets':[0.5,0.25,0.1,0.05],
 'matched_pair_max_relative_gap':0.02,'matched_target_relative_tolerance':0.2,
 'sigma_data':0.5,'P_mean':-1.2,'P_std':1.2,
 'polar':'CPU scipy.linalg.svd gesdd float64; tol=max(m,n)*eps64*smax',
 'matrix_selection':'all requires_grad two-dimensional parameters; no aspect scaling',
 'training':'fresh iid Gaussian X; true clean X targets, not Bayes teacher',
 'dtype':'float32 forward/backward, float64 SVD and diagnostics',
 'spectra':{'lognormal':{'beta':1},'powerlaw':{'a':1},'geometric':{'h':math.log(100)},'two_band':{'r':20,'fraction':0.5}},
 'source_commit':'94965885b5d94a8fcba0c3b6a31b895d3eb5deee',
}

def sha256(path):return hashlib.sha256(resolve_artifact(path).read_bytes()).hexdigest()

def finite_json(obj):
 if isinstance(obj,float):return math.isfinite(obj)
 if isinstance(obj,dict):return all(finite_json(v) for v in obj.values())
 if isinstance(obj,(list,tuple)):return all(finite_json(v) for v in obj)
 return True

def migration_entry(run,old_metadata,new_metadata):
 receipt_path=ROOT/'compatibility_receipt.json'
 if not artifact_exists(receipt_path) or run.parent.name!='search':return None
 receipt=read_json(receipt_path);key=run.relative_to(ROOT).as_posix()
 entry=receipt.get('reused_search_runs',{}).get(key) or receipt.get('resumed_search_runs',{}).get(key)
 if not entry or receipt['new_source_hashes']!=source_hashes():return None
 if old_metadata.get('source_hashes')!=entry['original_source_hashes']:return None
 if {k:v for k,v in old_metadata.items() if k!='source_hashes'}!={k:v for k,v in new_metadata.items() if k!='source_hashes'}:return None
 for relative,expected in entry['artifact_sha256'].items():
  if not artifact_exists(run/relative) or sha256(run/relative)!=expected:return None
 for relative,expected in entry.get('original_archive_artifacts',{}).items():
  archived=(ROOT/relative).resolve()
  if not archived.is_relative_to(ROOT) or not artifact_exists(archived) or sha256(archived)!=expected:return None
 return entry

def compatible_completed_search(run,old_metadata,new_metadata):
 entry=migration_entry(run,old_metadata,new_metadata)
 return bool(entry and artifact_exists(run/'complete.json') and
  read_json(run/'complete.json').get('source_hashes') in [source_hashes(),entry['original_source_hashes']])

def acquire_worker_lock():
 # The operating system releases this lock on process exit, including crashes.
 # PID enumeration can be incomplete across Windows sandbox boundaries.
 import msvcrt
 lock=open(ROOT/'worker.lock','a+b')
 lock.seek(0)
 if lock.read(1)==b'':lock.write(b'0');lock.flush()
 lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError as exc:
  lock.close();raise RuntimeError('Another nonlinear study worker already owns worker.lock') from exc
 write_json(ROOT/'worker_status.json',{'pid':os.getpid(),'status':'running','source_directory':str(SOURCE_ROOT),'started_unix':time.time()})
 return lock

def config():
 p=ROOT/'protocol.json'
 if not artifact_exists(p):write_json(p,DEFAULT)
 return read_json(p)

def seed_generator(seed,device='cpu'):
 return torch.Generator(device=device).manual_seed(int(seed))

def covariance(d,seed,family,device='cuda'):
 g=seed_generator(seed+1009)
 q,r=torch.linalg.qr(torch.randn(d,d,generator=g,dtype=torch.float64))
 q=q*torch.sign(torch.diag(r))[None,:]
 z=torch.randn(d,generator=g,dtype=torch.float64)
 j=torch.arange(1,d+1,dtype=torch.float64)
 if family=='lognormal':ell=torch.exp(z)
 elif family=='powerlaw':ell=1/j
 elif family=='geometric':ell=torch.exp(-math.log(100)*(j-1)/(d-1))
 elif family=='two_band':ell=torch.where(j<=d//2,20.,1.)
 else:raise ValueError(family)
 lam=torch.sort(ell,descending=True).values;lam/=lam.mean()
 return q.to(device=device,dtype=torch.float32),lam.to(device=device,dtype=torch.float32)

def make_model(d,seed,device='cuda'):
 torch.manual_seed(seed+2027)
 return Denoiser(d).to(device)

def batch(q,lam,n,g,sigma=None):
 d=len(lam);dev=lam.device
 x=(torch.randn(n,d,generator=g,device=dev)*lam.sqrt())@q.T
 if sigma is None:sig=torch.exp(-1.2+1.2*torch.randn(n,1,generator=g,device=dev))
 else:sig=torch.full((n,1),sigma,device=dev)
 y=x+sig*torch.randn(n,d,generator=g,device=dev)
 return x,y,sig

def weight(sig):return (sig.square()+0.25)/(sig*0.5).square()

def polar_numpy(g):
 a=np.asarray(g,dtype=np.float64)
 u,s,v=scipy.linalg.svd(a,full_matrices=False,lapack_driver='gesdd',check_finite=False)
 keep=s>(max(a.shape)*np.finfo(np.float64).eps*s[0])
 return np.asarray((u*keep)@v,dtype=np.float32),int(keep.sum())

class Updater:
 def __init__(self,model,method,workers):
  self.matrix=[(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim==2]
  self.aux=[(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim!=2]
  self.method=method;self.executor=ThreadPoolExecutor(workers) if method=='muon' else None
  self.last_ranks=[]
 @torch.no_grad()
 def step(self,lr,aux_lr):
  if self.method=='muon':
   gradients=[p.grad.detach().cpu().numpy() for _,p in self.matrix]
   result=list(self.executor.map(polar_numpy,gradients))
   self.last_ranks=[r for _,r in result]
   for (_,p),(delta,_) in zip(self.matrix,result):p.add_(torch.from_numpy(delta).to(p.device),alpha=-lr)
  else:
   for _,p in self.matrix:p.add_(p.grad,alpha=-lr)
  for _,p in self.aux:p.add_(p.grad,alpha=-aux_lr)
 def close(self):
  if self.executor:self.executor.shutdown()

def source_hashes():
 if not hasattr(source_hashes,'frozen'):
  source_hashes.frozen={name:sha256(ROOT/name if name=='protocol.json' else SOURCE_ROOT/name) for name in ['protocol.json','runner.py','model.py','runtime.py','summarize.py','storage.py']}
 return source_hashes.frozen

@torch.no_grad()
def integrated_excess(model,q,lam,n,seed):
 g=seed_generator(seed,lam.device.type);values=[];unweighted=[];mse=[]
 model.eval()
 for start in range(0,n,1024):
  x,y,sig=batch(q,lam,min(1024,n-start),g)
  dy=model(y,sig)
  bayes=((y@q)*(lam[None,:]/(lam[None,:]+sig.square())))@q.T
  e=(dy-bayes).square().mean(1)
  values.append((e*weight(sig).flatten()).double().cpu())
  unweighted.append(e.double().cpu());mse.append(((dy-x).square().mean(1)*weight(sig).flatten()).double().cpu())
 v=torch.cat(values);u=torch.cat(unweighted);m=torch.cat(mse)
 return {'weighted_bayes_excess_per_coordinate':v.mean().item(),
         'weighted_bayes_excess_mc_se':(v.std(unbiased=True)/math.sqrt(n)).item(),
         'bayes_excess_per_coordinate':u.mean().item(),
         'weighted_denoising_mse_per_coordinate':m.mean().item(),'n':n}

def checkpoint(model,g,q,lam,run,step,history,training_seconds,recovery=False):
 p=run/(f'recovery_{step:05d}.pt' if recovery else f'checkpoint_{step:05d}.pt')
 state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
             'training_rng_state':g.get_state().cpu(),'q':q.cpu(),'lambda':lam.cpu(),
             'step':step,'history':history,'training_seconds':training_seconds,
             'source_hashes':source_hashes()}
 write_artifact(p,lambda target:torch.save(state,target))
 return p

def train_one(cfg,family,seed,method,lr,steps,run,full_checkpoints,aux_lr=None):
 run.mkdir(parents=True,exist_ok=True)
 if aux_lr is None:aux_lr=lr
 metadata={'family':family,'seed':seed,'method':method,'matrix_lr':lr,'aux_lr':aux_lr,'planned_steps':steps,
  'batch_size':cfg['batch_size'],'validation_n':cfg['validation_n'],'full_checkpoints':full_checkpoints,'source_hashes':source_hashes()}
 meta_path=run/'run_metadata.json'
 migration=None
 if artifact_exists(meta_path) and read_json(meta_path)!=metadata:
  old_metadata=read_json(meta_path);migration=migration_entry(run,old_metadata,metadata)
  if migration is None:raise RuntimeError('Run metadata mismatch: '+str(run))
 if not artifact_exists(meta_path):write_json(meta_path,metadata)
 if artifact_exists(run/'complete.json'):
  completed=read_json(run/'complete.json')
  allowed=[source_hashes()]+([migration['original_source_hashes']] if migration else [])
  if completed.get('source_hashes') not in allowed:raise RuntimeError('Completed run source mismatch: '+str(run))
  return completed
 q,lam=covariance(cfg['dimension'],seed,family)
 model=make_model(cfg['dimension'],seed);g=seed_generator(seed+3011,'cuda')
 updater=Updater(model,method,cfg['polar_workers'])
 history=[];startstep=0;elapsed=0.
 existing=list_artifacts(run,'checkpoint_*.pt')+list_artifacts(run,'recovery_*.pt')
 if artifact_exists(run/'recovery.pt'):existing.append(run/'recovery.pt')
 if existing:
  restored=[(cp,torch.load(resolve_artifact(cp),map_location='cpu',weights_only=False)) for cp in existing]
  cp,old=max(restored,key=lambda item:item[1]['step'])
  if old['source_hashes']!=source_hashes():
   authorized=migration and old['source_hashes']==migration['original_source_hashes'] and migration['artifact_sha256'].get(cp.name)==sha256(cp)
   if not authorized:raise RuntimeError('Sources changed; cannot resume')
  if not isinstance(old['step'],int) or not 0<=old['step']<=steps:raise RuntimeError('Invalid checkpoint step')
  if not all(bool(torch.isfinite(v).all()) for v in old['model'].values()):raise RuntimeError('Nonfinite saved model')
  if not torch.equal(old['q'],q.cpu()) or not torch.equal(old['lambda'],lam.cpu()):raise RuntimeError('Checkpoint covariance mismatch')
  model.load_state_dict(old['model']);g.set_state(old['training_rng_state']);startstep=old['step'];history=old['history'];elapsed=old['training_seconds']
  if not finite_json(history):raise RuntimeError('Checkpoint history contains nonfinite validation metrics')
  resumed={'run':run.name,'resumed_from':cp.name,'checkpoint_sha256':sha256(cp),'step':startstep,'planned_steps':steps,
   'original_source_hashes':old['source_hashes'],'execution_source_hashes':source_hashes()}
  write_json(run/'resume_record.json',resumed);print(json.dumps(resumed),flush=True)
  write_json(run/'history.json',history)
 save_steps=(set(cfg['checkpoint_steps']+[steps]) | set(range(0,steps+1,cfg['checkpoint_interval']))) if full_checkpoints else {0,steps}
 def record(step):
  metrics=integrated_excess(model,q,lam,cfg['validation_n'],seed+4001)
  if not finite_json(metrics):
   write_json(run/'validation_failure.json',{'step':step,'reason':'nonfinite validation metric before checkpoint commit',
    'metrics_repr':{k:repr(v) for k,v in metrics.items()}})
   return False
  entry={'step':step,'training_seconds':elapsed,**metrics}
  history.append(entry)
  checkpoint(model,g,q,lam,run,step,history,elapsed)
  write_json(run/'history.json',history)
  print(json.dumps({'run':run.name,'family':family,'seed':seed,'method':method,'lr':lr,**entry}),flush=True)
  return True
 diverged=False;step=startstep;divergence_reason=None
 if not existing and not record(0):diverged=True;divergence_reason='nonfinite initial validation'
 try:
  for step in (range(startstep+1,steps+1) if not diverged else []):
   model.train();t=time.perf_counter()
   x,y,sig=batch(q,lam,cfg['batch_size'],g)
   model.zero_grad(set_to_none=True)
   loss=((model(y,sig)-x).square()*weight(sig)).mean()
   if not bool(torch.isfinite(loss)) or float(loss.detach())>1e12:
    diverged=True;divergence_reason='nonfinite or excessively large training loss';break
   loss.backward()
   if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]).all()):
    diverged=True;divergence_reason='nonfinite gradient';break
   updater.step(lr,aux_lr)
   torch.cuda.synchronize();elapsed+=time.perf_counter()-t
   if step in save_steps:
    if not record(step):diverged=True;divergence_reason='nonfinite validation before checkpoint commit';break
   elif step%cfg['recovery_interval']==0:
    checkpoint(model,g,q,lam,run,step,history,elapsed,recovery=True)
    print(json.dumps({'run':run.name,'step':step,'steps':steps,'train_loss':float(loss.detach()),'training_seconds':elapsed}),flush=True)
 except Exception as exc:
  write_json(run/'error.json',{'error':repr(exc),'step_attempted':step,'recovery':'resume from last fully saved checkpoint only'})
  raise
 finally:updater.close()
 result={'family':family,'seed':seed,'method':method,'lr':lr,'aux_lr':aux_lr,'steps':step,'planned_steps':steps,
         'diverged':diverged,'divergence_reason':divergence_reason,'final_validation':None if diverged else history[-1],
         'training_seconds':elapsed,'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'last_polar_ranks':updater.last_ranks,'checkpoint_count':len(list_artifacts(run,'checkpoint_*.pt')),'source_hashes':source_hashes()}
 write_json(run/'complete.json',result);return result

def search(cfg):
 rows=[];selection={}
 for family in FAMILIES:
  selection[family]={}
  for method in METHODS:
   candidates=[];grid=list(cfg[method+'_lrs']);pending=list(grid);round_number=0
   while True:
    aux_lr=None if method=='sgd' else selection[family]['sgd']['lr']
    for lr in pending:
     run=ROOT/'search'/f'{family}_{method}_lr{lr:g}'
     result=train_one(cfg,family,cfg['search_seed'],method,lr,cfg['search_steps'],run,False,aux_lr)
     rows.append(result)
     if not result['diverged']:candidates.append(result)
    if not candidates:raise RuntimeError('All candidates diverged: '+family+' '+method)
    best=min(candidates,key=lambda r:r['final_validation']['weighted_bayes_excess_per_coordinate'])
    selection[family][method]={'lr':best['lr'],'validation_excess':best['final_validation']['weighted_bayes_excess_per_coordinate'],
      'aux_lr':best['aux_lr'],'at_grid_boundary':best['lr'] in [min(grid),max(grid)],
      'search_seed':cfg['search_seed'],'grid':grid,'extension_rounds':round_number,'candidate_count':len(grid)}
    write_json(ROOT/'search_progress.json',{'completed':len(rows),'base_budget':sum(len(cfg[m+'_lrs']) for m in METHODS)*len(FAMILIES),
      'max_budget':len(FAMILIES)*sum(len(cfg[m+'_lrs'])+2*cfg['boundary_max_rounds'] for m in METHODS),'selection_so_far':selection})
    if not selection[family][method]['at_grid_boundary'] or round_number>=cfg['boundary_max_rounds']:break
    round_number+=1
    pending=[min(grid)/cfg['boundary_extension_factor'],max(grid)*cfg['boundary_extension_factor']]
    grid+=pending
 write_json(ROOT/'search_results.json',rows)
 write_json(ROOT/'lr_selection.json',{'protocol_sha256':sha256(ROOT/'protocol.json'),'selected':selection,'criterion':'final independent validation integrated weighted Bayes excess; no test data'})
 return selection

def main_runs(cfg):
 chosen=read_json(ROOT/'lr_selection.json')
 if chosen['protocol_sha256']!=sha256(ROOT/'protocol.json'):raise RuntimeError('LR selection protocol mismatch')
 rows=[]
 for family in FAMILIES:
  for seed in cfg['seeds']:
   for method in METHODS:
    lr=chosen['selected'][family][method]['lr']
    run=ROOT/'runs'/f'{family}_seed{seed}_{method}'
    rows.append(train_one(cfg,family,seed,method,lr,cfg['main_steps'],run,True,chosen['selected'][family]['sgd']['lr']))
    write_json(ROOT/'main_progress.json',{'completed':len(rows),'total':len(FAMILIES)*len(cfg['seeds'])*2,'runs':rows})
 return rows

@torch.no_grad()
def fixed_diagnostics(model,q,lam,sigma,cfg,seed):
 d=len(lam);c=lam.double().cpu().numpy()+sigma*sigma
 # Regress on whitened Gaussian input. Independent designs prevent evaluation leakage.
 # OLS is consistent for E[D Y^T] C^{-1}, and exact for a linear denoiser even at finite n.
 def sample_predictions(n,g):
  zz=[];dd=[]
  for start in range(0,n,1024):
   z=torch.randn(min(1024,n-start),d,generator=g,device=q.device)
   y=(z*(lam+sigma*sigma).sqrt())@q.T
   out=model(y,torch.full((len(z),1),sigma,device=q.device))@q
   zz.append(z.double().cpu().numpy());dd.append(out.double().cpu().numpy())
  return np.concatenate(zz),np.concatenate(dd)
 fits=[];target=np.diag(lam.double().cpu().numpy()/c)
 for k in range(2):
  z,dy=sample_predictions(cfg['fit_n_each'],seed_generator(seed+10007*k,'cuda'))
  # Control variate: project the residual to the known Bayes denoiser.
  residual_output=dy-(z*np.sqrt(c)[None,:])@target.T
  beta=scipy.linalg.solve(z.T@z,z.T@residual_output,assume_a='pos',check_finite=False)
  fits.append(target+beta.T/np.sqrt(c)[None,:])
 b=(fits[0]+fits[1])/2
 resid=b-target
 directional=(resid**2).sum(axis=0)
 fit_noise=((fits[0]-fits[1])**2).sum(axis=0)/4
 z,dy=sample_predictions(cfg['fixed_test_n'],seed_generator(seed+30011,'cuda'))
 yeig=z*np.sqrt(c)[None,:];bayes=yeig@target.T
 linear=yeig@b.T
 e=dy-bayes;nonlinear=dy-linear;le=linear-bayes
 n=len(z)
 weighted_directional=c*directional
 rows={'sigma':sigma,'kappa_C':float(c.max()/c.min()),
  'bayes_excess_per_coordinate':float(np.mean(e**2)),
  'bayes_excess_mc_se':float(np.std(np.mean(e**2,axis=1),ddof=1)/math.sqrt(n)),
  'linear_excess_population_per_coordinate':float(weighted_directional.mean()),
  'high_half_absolute_linear_excess_per_coordinate':float(weighted_directional[:d//2].sum()/d),
  'low_half_absolute_linear_excess_per_coordinate':float(weighted_directional[d//2:].sum()/d),
  'high_half_linear_fit_noise_per_coordinate':float((c*fit_noise)[:d//2].sum()/d),
  'low_half_linear_fit_noise_per_coordinate':float((c*fit_noise)[d//2:].sum()/d),
  'linear_excess_fit_noise_estimate':float(np.mean(c*fit_noise)),
  'linear_excess_test_per_coordinate':float(np.mean(le**2)),
  'nonlinear_residual_test_per_coordinate':float(np.mean(nonlinear**2)),
  'cross_term_test_per_coordinate':float(2*np.mean(le*nonlinear)),
  'decomposition_identity_error':float(abs(np.mean(e**2)-np.mean(le**2)-np.mean(nonlinear**2)-2*np.mean(le*nonlinear))),
  'high_half_linear_error_fraction':float(weighted_directional[:d//2].sum()/max(weighted_directional.sum(),1e-300)),
  'mean_output_squared_per_coordinate':float(np.mean(np.mean(dy,axis=0)**2)),
  'fit_n_each':cfg['fit_n_each'],'test_n':n}
 return rows,{'Wlin_eigenbasis':b,'directional_error':directional,'directional_fit_noise':fit_noise,'lambda':lam.cpu().numpy(),'c':c,'weighted_directional_error':weighted_directional}

def select_matched_checkpoints(cfg):
 rows=[]
 for family in FAMILIES:
  for seed in cfg['seeds']:
   histories={}
   for method in METHODS:
    run=ROOT/'runs'/f'{family}_seed{seed}_{method}';p=run/'history.json'
    try:raw=read_json(p)
    except (OSError,json.JSONDecodeError):raw=[]
    if not isinstance(raw,list):raw=[]
    histories[method]=[r for r in raw if isinstance(r,dict) and finite_json(r)
     and isinstance(r.get('step'),int) and r.get('weighted_bayes_excess_per_coordinate',0)>0
     and artifact_exists(run/f"checkpoint_{r['step']:05d}.pt")]
   initial=next((r for r in histories['sgd'] if r['step']==0),None)
   f0=initial['weighted_bayes_excess_per_coordinate'] if initial else None
   for fraction in cfg['matched_loss_targets']:
    if f0 is None or not all(histories.values()):
     rows.append({'family':family,'seed':seed,'relative_target':fraction,'matched':False,
      'reason':'Missing, empty, nonfinite, nonpositive, or checkpoint-inconsistent validation history'})
     continue
    target=f0*fraction;candidates=[]
    for a in histories['sgd']:
     for b in histories['muon']:
      av=a['weighted_bayes_excess_per_coordinate'];bv=b['weighted_bayes_excess_per_coordinate']
      gap=2*abs(av-bv)/(av+bv)
      targetgap=max(abs(av/target-1),abs(bv/target-1))
      if a['step']>0 and b['step']>0 and gap<=cfg['matched_pair_max_relative_gap'] and targetgap<=cfg['matched_target_relative_tolerance']:
       candidates.append((abs(math.log((av+bv)/(2*target))),gap,a,b))
    row={'family':family,'seed':seed,'relative_target':fraction,'target_validation_excess':target,'matched':bool(candidates)}
    if candidates:
     _,gap,a,b=min(candidates,key=lambda item:(item[0],item[1],item[2]['step']+item[3]['step']))
     row.update({'sgd_step':a['step'],'muon_step':b['step'],'sgd_validation_excess':a['weighted_bayes_excess_per_coordinate'],
      'muon_validation_excess':b['weighted_bayes_excess_per_coordinate'],'validation_relative_gap':gap})
    else:row['reason']='No stored validation checkpoint pair satisfies both pair and target tolerances'
    rows.append(row)
 write_json(ROOT/'matched_checkpoint_selection.json',{'selection_basis':'validation only, before test evaluation','pairs':rows})
 return rows

def evaluate(cfg):
 if not artifact_exists(ROOT/'lr_selection.json'):raise RuntimeError('Freeze LR selection before accessing test data')
 matched=select_matched_checkpoints(cfg)
 allrows=[]
 for family in FAMILIES:
  for seed in cfg['seeds']:
   for method in METHODS:
    run=ROOT/'runs'/f'{family}_seed{seed}_{method}'
    if not artifact_exists(run/'complete.json'):continue
    q,lam=covariance(cfg['dimension'],seed,family);model=make_model(cfg['dimension'],seed)
    diagnostic_steps=set(cfg['checkpoint_steps'])
    diagnostic_steps.update(r[method+'_step'] for r in matched if r['matched'] and r['family']==family and r['seed']==seed)
    rows=[]
    for cp in list_artifacts(run,'checkpoint_*.pt'):
     outpath=run/f'test_{cp.stem.split("_")[-1]}.json'
     saved=torch.load(resolve_artifact(cp),map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval();step=saved['step']
     meta={'family':family,'seed':seed,'method':method,'step':step,'training_seconds':saved['training_seconds'],'checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha256(cp)}
     try:cached=read_json(outpath)
     except (OSError,json.JSONDecodeError):cached=[]
     if not isinstance(cached,list):cached=[]
     cached=[r for r in cached if isinstance(r,dict) and finite_json(r) and all(r.get(k)==v for k,v in meta.items())]
     integrated_row=next((r for r in cached if r.get('kind')=='integrated' and 'weighted_bayes_excess_per_coordinate' in r),None)
     if integrated_row is None:
      integrated=integrated_excess(model,q,lam,cfg['test_n'],seed+5003)
      if finite_json(integrated):integrated_row={**meta,'kind':'integrated',**integrated}
      else:write_json(run/f'test_failure_{step:05d}.json',{'step':step,'reason':'nonfinite integrated test metrics'})
     cp_rows=[integrated_row] if integrated_row else []
     arrays={}
     array_path=run/f'diagnostics_{step:05d}.npz'
     if cached and artifact_exists(array_path):
      try:
       with np.load(resolve_artifact(array_path)) as archive:arrays={k:archive[k] for k in archive.files}
      except (ValueError,OSError,EOFError,zipfile.BadZipFile):arrays={}
     for k,sigma in enumerate(cfg['fixed_sigmas']):
      fixed_row=next((r for r in cached if r.get('kind')=='fixed_sigma' and r.get('sigma')==sigma),None)
      required_keys=['Wlin_eigenbasis','directional_error','directional_fit_noise','lambda','c','weighted_directional_error']
      arrays_ok=all(f'sigma{k}_{key}' in arrays and np.all(np.isfinite(arrays[f'sigma{k}_{key}'])) for key in required_keys)
      if fixed_row and arrays_ok:cp_rows.append(fixed_row)
      elif step in diagnostic_steps:
       metrics,a=fixed_diagnostics(model,q,lam,sigma,cfg,seed+6007+409*k)
       if finite_json(metrics) and all(np.all(np.isfinite(val)) for val in a.values()):
        cp_rows.append({**meta,'kind':'fixed_sigma',**metrics});arrays.update({f'sigma{k}_{key}':val for key,val in a.items()})
       else:write_json(run/f'test_failure_{step:05d}_sigma{k}.json',{'step':step,'sigma':sigma,'reason':'nonfinite fixed-noise test diagnostics'})
     if arrays:write_artifact(run/f'diagnostics_{step:05d}.npz',lambda target:np.savez_compressed(target,**arrays))
     write_json(outpath,cp_rows);rows.extend(cp_rows)
     print(json.dumps({'evaluated':str(outpath.relative_to(ROOT)),'test_excess':integrated_row['weighted_bayes_excess_per_coordinate'] if integrated_row else None}),flush=True)
    write_json(run/'test_metrics.json',rows);allrows.extend(rows)
 write_json(ROOT/'test_metrics.json',allrows)
 return allrows

def tests():
 rng=np.random.default_rng(137);checks={}
 for shape,rank in [((12,7),4),((7,12),4),((8,8),8),((8,8),0)]:
  a=rng.normal(size=(shape[0],rank))@rng.normal(size=(rank,shape[1]))
  p,r=polar_numpy(a)
  u,s,v=scipy.linalg.svd(a,full_matrices=False)
  ref=u[:,:rank]@v[:rank]
  err=float(np.linalg.norm(p-ref));checks[f'polar_{shape}_{rank}']={'error':err,'numerical_rank':r}
  assert r==rank and err<2e-6
 model=make_model(8,137,'cpu');other=copy.deepcopy(model)
 x=torch.randn(32,8);sig=torch.full((32,1),.5)
 for m in [model,other]:m(x,sig).square().mean().backward()
 updater=Updater(model,'sgd',1);updater.step(.01,.01)
 with torch.no_grad():
  for p in other.parameters():
   if p.requires_grad:p.add_(p.grad,alpha=-.01)
 diff=max((a-b).abs().max().item() for a,b in zip(model.parameters(),other.parameters()))
 assert diff==0;checks['sgd_exact_update']=diff
 d=8;n=1024;z=rng.normal(size=(n,d));b=rng.normal(size=(d,d));dy=z@b.T
 bhat=scipy.linalg.solve(z.T@z,z.T@dy).T
 fit_err=float(np.linalg.norm(bhat-b));assert fit_err<1e-12;checks['linear_fit_exact_for_linear_function']=fit_err
 q,lam=covariance(128,17,'lognormal','cpu');checks['covariance_mean_eigenvalue']=float(lam.mean());assert abs(float(lam.mean())-1)<1e-6
 ma=make_model(128,17,'cpu');mb=make_model(128,17,'cpu')
 assert all(torch.equal(a,b) for a,b in zip(ma.state_dict().values(),mb.state_dict().values()))
 ba=batch(q,lam,32,seed_generator(17+3011));bb=batch(q,lam,32,seed_generator(17+3011))
 assert all(torch.equal(a,b) for a,b in zip(ba,bb));checks['paired_initialization_and_minibatches']=True
 checks['passed']=True;write_json(ROOT/'runner_tests.json',checks);print(json.dumps(checks,indent=2))

def benchmark(cfg):
 rows=[]
 for method in METHODS:
  small=copy.deepcopy(cfg);small['validation_n']=128;small['checkpoint_steps']=[0,50]
  run=ROOT/'benchmark_runs'/method
  res=train_one(small,'lognormal',137,method,.01,50,run,True,.01)
  rows.append(res)
 write_json(ROOT/'benchmark_training.json',rows)
 print(json.dumps(rows,indent=2))

def snapshot(cfg):
 files=['runner.py','model.py','runtime.py','summarize.py','storage.py','protocol.json','upstream_diffusion_nn_lib.py','upstream_diffusion_edm_lib.py']
 write_json(ROOT/'provenance.json',{'files':{name:sha256(ROOT/name if name=='protocol.json' else SOURCE_ROOT/name) for name in files},'executed_source_directory':str(SOURCE_ROOT),
  'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
  'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'platform':platform.platform(),
  'primary_sources':['https://arxiv.org/html/2503.03206v3#A9.SS2','https://arxiv.org/html/2503.03206v3#A9.SS4',
   'https://github.com/Animadversio/DiffusionLearningCurve/blob/'+cfg['source_commit']+'/core/diffusion_nn_lib.py',
   'https://github.com/Animadversio/DiffusionLearningCurve/blob/'+cfg['source_commit']+'/core/diffusion_edm_lib.py'],
  'wp_replication_scope':'backbone and EDM architecture matched; optimizers, population samples, seed/spectra/search/evaluations are this study choices',
  'independent_model_validation':read_json(ROOT/'model_validation.json') if artifact_exists(ROOT/'model_validation.json') else None})

def test_coverage(cfg,metrics):
 keys={(r['family'],r['seed'],r['method'],r['step'],r['kind'],r.get('sigma')) for r in metrics if finite_json(r)}
 matched=read_json(ROOT/'matched_checkpoint_selection.json')['pairs']
 issues=[]
 for family in FAMILIES:
  for seed in cfg['seeds']:
   for method in METHODS:
    run=ROOT/'runs'/f'{family}_seed{seed}_{method}'
    steps={int(cp.stem.split('_')[-1]) for cp in list_artifacts(run,'checkpoint_*.pt')}
    expected_steps=(set(cfg['checkpoint_steps']) | set(range(0,cfg['main_steps']+1,cfg['checkpoint_interval'])) | {cfg['main_steps']})
    expected_steps={step for step in expected_steps if step<=cfg['main_steps']}
    diagnostic_steps=set(cfg['checkpoint_steps'])
    diagnostic_steps.update(p[method+'_step'] for p in matched if p['matched'] and p['family']==family and p['seed']==seed)
    for step in sorted(steps | expected_steps):
     if step not in steps:issues.append({'family':family,'seed':seed,'method':method,'step':step,'reason':'missing planned parameter checkpoint'})
     required=[('integrated',None)]
     if step in diagnostic_steps:required += [('fixed_sigma',s) for s in cfg['fixed_sigmas']]
     for kind,sigma in required:
      if (family,seed,method,step,kind,sigma) not in keys:
       issues.append({'family':family,'seed':seed,'method':method,'step':step,'kind':kind,'sigma':sigma,'reason':'missing finite test metrics'})
 return {'complete':not issues,'issues':issues,'matched_pairs':sum(p['matched'] for p in matched)}

def run_action(action):
 cfg=config();snapshot(cfg)
 if action=='test':tests()
 elif action=='benchmark':benchmark(cfg)
 elif action=='search':search(cfg)
 elif action=='main':main_runs(cfg)
 elif action=='evaluate':evaluate(cfg)
 else:
  tests()
  if not artifact_exists(ROOT/'lr_selection.json'):search(cfg)
  runs=main_runs(cfg);metrics=evaluate(cfg)
  from summarize import summarize
  summary=summarize(ROOT)
  failed=[r for r in runs if r['diverged'] or r['steps']!=r['planned_steps']]
  coverage=test_coverage(cfg,metrics)
  write_json(ROOT/'study_complete.json',{'status':'complete' if not failed and coverage['complete'] and summary['diagnostic_coverage_issue_count']==0 else 'completed_with_unresolved_cases',
    'attempted_main_runs':len(runs),'full_budget_completed':len(runs)-len(failed),'unresolved_runs':failed,
    'evaluated_checkpoint_count':sum(r['kind']=='integrated' for r in metrics),'metric_row_count':len(metrics),
    'protocol_sha256':sha256(ROOT/'protocol.json'),'summary':summary,'test_coverage':coverage})
  write_json(ROOT/'worker_status.json',{'pid':os.getpid(),'status':'finished','source_directory':str(SOURCE_ROOT),'finished_unix':time.time()})

def execute_with_failure_status(action,worker_lock):
 try:run_action(action)
 except BaseException as exc:
  if worker_lock:
   import traceback
   write_json(ROOT/'worker_status.json',{'pid':os.getpid(),'status':'failed','source_directory':str(SOURCE_ROOT),
    'failed_unix':time.time(),'error':repr(exc),'traceback':traceback.format_exc()})
  raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['test','benchmark','search','main','evaluate','all']);args=ap.parse_args()
 _worker_lock=acquire_worker_lock() if args.action in ['search','main','evaluate','all'] else None
 execute_with_failure_status(args.action,_worker_lock)
