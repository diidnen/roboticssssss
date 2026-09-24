"""Read-only diagnosis of dropped aleatoric uncertainty; no planner changes."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

ROOT=Path('/home/exouser/FORTE')
DATA=ROOT/'analysis/results/current_contract_physical_belief_20260905'
OUT=ROOT/'analysis/results/current_contract_restore_and_matched_pilot_20260905'


def main():
    with (DATA/'PHYSICAL_BELIEF_PREDICTIONS.csv').open() as f:rows=list(csv.DictReader(f))
    details=[]
    for r in rows:
        mu=np.array([float(r[f'mu_{i}']) for i in range(3)])
        sd=np.array([float(r[f'sigma_{i}']) for i in range(3)])
        y=float(r['target_mu']);lower,upper=mu.min(),mu.max()
        epistemic=float(np.var(mu));aleatoric=float(np.square(sd).mean())
        # Analytic Gaussian mixture moments. No truncation/recalibration.
        mixture_var=epistemic+aleatoric
        lo_cdf=float(norm.cdf(lower,mu,sd).mean());hi_cdf=float(norm.cdf(upper,mu,sd).mean())
        quadrature_errors={}
        for n in (16,32):
            nodes,weights=np.polynomial.hermite.hermgauss(n)
            samples=mu[:,None]+np.sqrt(2)*sd[:,None]*nodes
            weights=np.tile(weights[None]/np.sqrt(np.pi)/3,(3,1))
            qmean=float((samples*weights).sum());qvar=float((weights*(samples-qmean)**2).sum())
            quadrature_errors[n]=max(abs(qmean-mu.mean()),abs(qvar-mixture_var))
        assert max(quadrature_errors.values())<1e-12
        details.append({'context_id':r['context_id'],'split':r['split'],'root_seed':r['root_seed'],
            'target_mu_analysis_only':y,'mean':float(mu.mean()),'empirical_support_min':lower,'empirical_support_max':upper,
            'empirical_support_distribution_std':float(np.sqrt(epistemic)),
            'aleatoric_std':float(np.sqrt(aleatoric)),'gaussian_mixture_std':float(np.sqrt(mixture_var)),
            'variance_fraction_omitted_by_member_mean_only_support':aleatoric/mixture_var,
            'gaussian_probability_mass_inside_member_mean_span':hi_cdf-lo_cdf,
            'gaussian_probability_negative_mu':float(norm.cdf(0,mu,sd).mean()),
            'truth_inside_member_mean_span':bool(lower<=y<=upper),
            'gaussian_mixture_pit':float(norm.cdf(y,mu,sd).mean()),
            'gaussian_mixture_nll':float(-(logsumexp(norm.logpdf(y,mu,sd))-np.log(3))),
            'quadrature_moment_max_error':max(quadrature_errors.values())})
    summary={}
    for split in ('TRAIN','VAL','TEST','DIAGNOSTIC'):
        selected=[r for r in details if r['split']==split]
        summary[split]={'count':len(selected),'independent_root_count':len({r['root_seed'] for r in selected}),
            **{k:float(np.mean([r[k] for r in selected])) for k in (
                'variance_fraction_omitted_by_member_mean_only_support','gaussian_probability_mass_inside_member_mean_span',
                'gaussian_probability_negative_mu','truth_inside_member_mean_span','gaussian_mixture_nll')}}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'POSTERIOR_INTEGRATION_DIAGNOSTIC.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    result={'scope':'offline diagnostics only; not a deployed replacement posterior',
        'source_predictions':str(DATA/'PHYSICAL_BELIEF_PREDICTIONS.csv'),
        'source_sha256':hashlib.sha256((DATA/'PHYSICAL_BELIEF_PREDICTIONS.csv').read_bytes()).hexdigest(),
        'current_planner_representation':'three member means at weight 1/3; member sigma ignored',
        'density_diagnostic':'equal-weight Gaussian mixture including trained sigma heads; untruncated',
        'mathematical_quadrature_moment_parity':True,'calibration_fitted':False,'training_performed':False,
        'posterior_interface_changed':False,'runtime_deployed':False,
        'conclusion':'Point discrimination is improved; mean-only integration discards most predictive variance. Gaussian mixture includes unphysical negative-mu mass, so do not silently deploy it without an explicit support contract.',
        'summary':summary}
    (OUT/'POSTERIOR_INTEGRATION_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
