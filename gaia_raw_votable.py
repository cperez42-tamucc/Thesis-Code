#!/usr/bin/env python3
"""Convert a Gaia DR3 TAP VOTable into Galactocentric SVGPR inputs.

Order: X=[z (kpc), R (kpc), phi (radians relative to Sun)];
       V=[v_z, v_R, v_phi] (km/s); Cobs=[N,3,3] measurement covariance.

Uses first-order propagation of all five correlated Gaia astrometric errors
and independent radial-velocity errors. This is not a Bayesian distance model.
The VOTable supplied in the associated TAP query lacks grvs_mag and
rv_template_teff, so paper-specific radial-velocity corrections cannot be made.
"""
from __future__ import annotations
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

LOG = logging.getLogger('gaia_benchmark')
FIELDS = (
    'source_id','ra','dec','parallax','pmra','pmdec','radial_velocity',
    'ra_error','dec_error','parallax_error','pmra_error','pmdec_error',
    'radial_velocity_error',
    'ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr',
    'dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr',
    'parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr',
    'parallax_over_error','ruwe','astrometric_params_solved',
)
# Raw astrometric covariance order (ra*, dec, parallax, pmra*, pmdec).
CORRS = (
    (0,1,'ra_dec_corr'),(0,2,'ra_parallax_corr'),
    (0,3,'ra_pmra_corr'),(0,4,'ra_pmdec_corr'),
    (1,2,'dec_parallax_corr'),(1,3,'dec_pmra_corr'),
    (1,4,'dec_pmdec_corr'),(2,3,'parallax_pmra_corr'),
    (2,4,'parallax_pmdec_corr'),(3,4,'pmra_pmdec_corr'),
)


def _astropy_orientation():
    """Obtain ICRS->Galactocentric rigid rotation from the public Astropy API."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric
    # Astropy Galactocentric has Sun at negative X. At Sun, v_R=-U and
    # the left-handed positive v_phi convention is +V.
    frame = Galactocentric(
        galcen_distance=8.277*u.kpc, z_sun=0.0208*u.kpc,
        galcen_v_sun=np.array([9.3, 251.5, 8.59])*(u.km/u.s),
    )
    # 0-distance ICRS origin + three 1-kpc Cartesian basis directions.
    reference = SkyCoord(
        ra=np.array([0.,0.,90.,0.])*u.deg,
        dec=np.array([0.,0.,0.,90.])*u.deg,
        distance=np.array([0.,1.,1.,1.])*u.kpc,frame='icrs',
    ).transform_to(frame)
    xyz = np.stack([reference.x.to_value(u.kpc),
                    reference.y.to_value(u.kpc),
                    reference.z.to_value(u.kpc)],axis=-1)
    offset = xyz[0]
    rotation = (xyz[1:]-offset).T
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-10):
        raise RuntimeError('Astropy coordinate transform did not produce a rigid rotation')
    if offset[0] > -7.0:
        raise RuntimeError('Unexpected Astropy Galactocentric X-axis sign; audit conventions')
    solar_phi=float(np.arctan2(offset[1],offset[0]))
    return rotation,offset,solar_phi,np.array([9.3,251.5,8.59])


def _normalize_field_name(name):
    """Match VOTable FIELD name/ID despite case, punctuation or generated IDs."""
    return ''.join(c for c in str(name).casefold() if c.isalnum())


def _resolve_columns(table):
    """Map canonical Gaia ADQL field names to actual Astropy record-array names.

    VOTable FIELD names, IDs and numpy dtype field names are not guaranteed
    to coincide, especially when served by TAP systems or Astropy versions.
    """
    rows = table.array
    dtype_names = tuple(rows.dtype.names or ())
    aliases = {}
    for actual in dtype_names:
        aliases.setdefault(_normalize_field_name(actual), actual)
    for position, field in enumerate(table.fields):
        if position >= len(dtype_names):
            break
        actual = dtype_names[position]
        for label in (getattr(field, 'name', None), getattr(field, 'ID', None)):
            if label:
                aliases.setdefault(_normalize_field_name(label), actual)
    return {canonical: aliases[_normalize_field_name(canonical)]
            for canonical in FIELDS if _normalize_field_name(canonical) in aliases}


def _describe_table(table):
    dtype_names = tuple(table.array.dtype.names or ())
    details=[]
    for i, field in enumerate(table.fields):
        details.append({'dtype':dtype_names[i] if i < len(dtype_names) else None,
                        'name':getattr(field,'name',None),
                        'ID':getattr(field,'ID',None)})
    return details


def _read_numeric_block(rows, start, stop, names, column_map):
    values={}
    for name in names:
        field=rows[column_map[name]][start:stop]
        # Gaia columns such as astrometric_params_solved can be masked int16.
        # Integer masked arrays cannot be filled with floating-point NaN.
        # Cast BEFORE filling, retaining missing-value masks as NaNs.
        if np.ma.isMaskedArray(field):
            values[name]=np.ma.asarray(field).astype(np.float64).filled(np.nan)
        else:
            values[name]=np.asarray(field,dtype=np.float64)
    return values


# Cylindrical radial velocity and its Jacobian are undefined at R=0.
# Exclude individual ill-conditioned objects, never abort an entire Gaia chunk.
MIN_CYLINDRICAL_R_KPC = 0.01

def _galactocentric_radius(d, rotation, offset):
    """Return cylindrical R for Gaia astrometry, before velocity conversion.

    Uses the same inverse-parallax distances and ICRS->Galactocentric
    orientation as _convert_chunk, so the exclusion rule is consistent.
    """
    alpha = np.deg2rad(d['ra'])
    delta = np.deg2rad(d['dec'])
    cd = np.cos(delta)
    direction = np.stack((cd * np.cos(alpha),
                          cd * np.sin(alpha),
                          np.sin(delta)), axis=1)
    gal_xy = ((direction / d['parallax'][:, None]) @ rotation[:2, :].T
              + offset[None, :2])
    return np.hypot(gal_xy[:, 0], gal_xy[:, 1])

def _convert_chunk(d, rotation, offset, sun_phi, sun_vel):
    """Vectorized exact mean transformation and linear covariance propagation.

    Returns X[N,3], V[N,3], Cobs[N,3,3], all physical units.
    Astrometry: use inverse-parallax distance (kpc = 1 / parallax_mas).
    """
    n=len(d['ra'])
    alpha=np.deg2rad(d['ra']); delta=np.deg2rad(d['dec'])
    ca,sa=np.cos(alpha),np.sin(alpha)
    cd,sd=np.cos(delta),np.sin(delta)
    p=d['parallax']
    distance=1/p
    unit=np.stack((cd*ca,cd*sa,sd),axis=1)
    pa=np.stack((-sa,ca,np.zeros(n)),axis=1)
    pd=np.stack((-sd*ca,-sd*sa,cd),axis=1)
    tangent=np.stack((ca,sa,np.zeros(n)),axis=1)
    tangential=4.740470463533349 * distance
    icrs_position=distance[:,None]*unit
    icrs_velocity=d['radial_velocity'][:,None]*unit + tangential[:,None]*(
        d['pmra'][:,None]*pa+d['pmdec'][:,None]*pd)
    xyz=icrs_position@rotation.T+offset
    vcart=icrs_velocity@rotation.T+sun_vel
    xx,yy,zz=xyz.T
    vx,vy,vz=vcart.T
    R=np.hypot(xx,yy)
    if np.any(~np.isfinite(R) | (R < MIN_CYLINDRICAL_R_KPC)):
        raise ValueError('A source is too near Galactic center for stable cylindrical conversion')
    phi=np.arctan2(np.sin(np.arctan2(yy,xx)-sun_phi),
                   np.cos(np.arctan2(yy,xx)-sun_phi))
    vR=(xx*vx+yy*vy)/R
    vphi=(yy*vx-xx*vy)/R   # left-handed: solar V yields positive v_phi
    X=np.stack((zz,R,phi),axis=1)
    V=np.stack((vz,vR,vphi),axis=1)

    # 6 observational coordinates: alpha [rad], delta [rad], parallax [mas],
    # pmra/pmdec [mas/yr], radial_velocity [km/s].  Astrometric covariance
    # includes all supplied Gaia pairwise correlations; RV is independent.
    dv=np.zeros((n,6,3),dtype=np.float64)
    dr=np.zeros((n,6,3),dtype=np.float64)
    k=4.740470463533349
    dv[:,0,:]=d['radial_velocity'][:,None]*cd[:,None]*pa + tangential[:,None]*(
        -d['pmra'][:,None]*tangent-d['pmdec'][:,None]*sd[:,None]*pa)
    dr[:,0,:]=distance[:,None]*cd[:,None]*pa
    dv[:,1,:]=d['radial_velocity'][:,None]*pd - tangential[:,None]*d['pmdec'][:,None]*unit
    dr[:,1,:]=distance[:,None]*pd
    dv[:,2,:]=(-k/(p*p))[:,None]*(d['pmra'][:,None]*pa+d['pmdec'][:,None]*pd)
    dr[:,2,:]=(-1/(p*p))[:,None]*unit
    dv[:,3,:]=tangential[:,None]*pa
    dv[:,4,:]=tangential[:,None]*pd
    dv[:,5,:]=unit
    dv=dv@rotation.T
    dr=dr@rotation.T
    jac=np.zeros((n,3,6),dtype=np.float64)
    for j in range(6):
        dx,dy=dr[:,j,0],dr[:,j,1]
        dwx,dwy,dwz=dv[:,j,0],dv[:,j,1],dv[:,j,2]
        dR=(xx*dx+yy*dy)/R
        jac[:,0,j]=dwz
        jac[:,1,j]=(dx*vx+xx*dwx+dy*vy+yy*dwy-vR*dR)/R
        jac[:,2,j]=(dy*vx+yy*dwx-dx*vy-xx*dwy-vphi*dR)/R

    mas_rad=np.deg2rad(1/3_600_000.)
    # Gaia ra_error is the tangent-plane alpha* error, not delta-alpha itself.
    sig=np.stack((d['ra_error']*mas_rad/np.maximum(np.abs(cd),1e-4),
                  d['dec_error']*mas_rad,
                  d['parallax_error'],d['pmra_error'],d['pmdec_error'],
                  d['radial_velocity_error']),axis=1)
    corr=np.broadcast_to(np.eye(6,dtype=np.float64),(n,6,6)).copy()
    for i,j,name in CORRS:
        corr[:,i,j]=corr[:,j,i]=d[name]
    # Not available from Gaia TAP query: correlations of RV with astrometry.
    raw_cov=corr*sig[:,:,None]*sig[:,None,:]
    # Efficient propagation using numpy batched linear algebra.
    Cobs=np.matmul(np.matmul(jac,raw_cov),jac.transpose(0,2,1))
    Cobs=(Cobs+Cobs.transpose(0,2,1))*.5
    # Prevent negative eigenvalues caused by precision/rounding; no fake noise.
    smallest=np.linalg.eigvalsh(Cobs)[:,0]
    add=np.maximum(-smallest,0)+1e-8
    Cobs[:,range(3),range(3)]+=add[:,None]
    return X.astype(np.float32),V.astype(np.float32),Cobs.astype(np.float32)


def prepare_votable(vot_file:Path, cache:Path, *, min_snr:float=5.0,
                     max_ruwe:float=1.4, chunk_rows:int=65536,
                     force:bool=False) -> Path:
    """One-time ingestion of raw .vot/.vot.gz file into validated local NPZ.

    Never silently reuse a cache from a different raw file, filter or version.
    Writes an atomic cache plus a provenance sidecar. This step can need
    tens of GB of RAM and several GB of free disk for the full Gaia RVS.
    """
    from astropy.io.votable import parse_single_table
    vot_file=Path(vot_file).expanduser().resolve()
    cache=Path(cache).expanduser().resolve()
    if not vot_file.is_file():
        raise FileNotFoundError(f'Gaia raw VOTable does not exist: {vot_file}')
    if chunk_rows<1: raise ValueError('chunk_rows must be positive')
    cache.parent.mkdir(parents=True,exist_ok=True)
    sidecar=cache.with_suffix('.provenance.json')
    stats=vot_file.stat()
    expected=dict(source=str(vot_file),source_bytes=stats.st_size,
                  source_mtime_ns=stats.st_mtime_ns,min_snr=min_snr,
                  max_ruwe=max_ruwe,converter_version='galactocentric-v5-radius-filter')
    if not force and cache.is_file() and sidecar.is_file():
        old=json.loads(sidecar.read_text())
        if all(old.get(k)==v for k,v in expected.items()):
            LOG.info('Reusing validated raw Gaia cache: %s',cache)
            return cache
    LOG.info('Reading Gaia raw VOTable %s; this is a one-time conversion.',vot_file)
    # The VOTable is not attached to this ChatGPT conversation. Its actual
    # table schema is verified at runtime against the ADQL query fields.
    vt=parse_single_table(str(vot_file),invalid='mask',verify='warn',
                          chunk_size=8192)
    rows=vt.array
    column_map=_resolve_columns(vt)
    missing=set(FIELDS)-{'source_id'}-set(column_map)
    if missing:
        raise ValueError(
            f'Raw Gaia VOTable missing REQUIRED measurement columns: {sorted(missing)}. '
            f'Available VOTable FIELD names/IDs and Astropy dtype names: {_describe_table(vt)}. '
            'Check that the file is the actual Gaia TAP result, not a query metadata/error VOTable.')
    has_source_id=('source_id' in column_map)
    if not has_source_id:
        LOG.warning('VOTable does not contain a source_id column (after checking FIELD '
                    'names, IDs and dtype names). It is optional for GP fitting; '
                    'the cache will omit source_id rather than invent Gaia identifiers. '
                    'Confirm the file is your intended Gaia result.')
    LOG.info('Resolved Gaia columns (canonical -> VOTable dtype): %s', column_map)
    rotation,offset,sun_phi,sun_vel=_astropy_orientation()
    N=len(rows)
    LOG.info('VOTable rows=%d; allocating conversion buffers (~%0.1f GiB)',
             N,N*(3+3+9)*4/(1024**3))
    # RAM: output buffers are float32 and later written to cache atomically.
    X=np.empty((N,3),dtype=np.float32)
    V=np.empty((N,3),dtype=np.float32)
    Cobs=np.empty((N,3,3),dtype=np.float32)
    ids=np.empty(N,dtype=np.int64) if has_source_id else None
    count=0; rejected=0; rejected_near_center=0
    numerical=[f for f in FIELDS if f != 'source_id']
    for start in range(0,N,chunk_rows):
        stop=min(start+chunk_rows,N)
        d=_read_numeric_block(rows,start,stop,numerical,column_map)
        nonnull=np.ones(stop-start,dtype=bool)
        for f in numerical:
            nonnull &= np.isfinite(d[f])
        good=nonnull & (d['parallax']>0) & (d['parallax_error']>0) & (
            d['parallax_over_error']>=min_snr) & (d['ruwe']<=max_ruwe) & (np.isin(d['astrometric_params_solved'],[31,95]))
        # Already required in query: 5p/6p solution and available radial velocity.
        for f in ('ra_error','dec_error','pmra_error','pmdec_error','radial_velocity_error'):
            good &= d[f]>=0
        for _,_,f in CORRS:
            good &= np.abs(d[f])<=1.000001
        ng=int(good.sum());rejected+=len(good)-ng
        if ng:
            cut={f:values[good] for f,values in d.items()}
            # A few distant sightlines can pass through R ~ 0 kpc. Their
            # cylindrical velocity and derivatives are poorly defined.
            # Drop just those rows instead of failing the entire catalogue.
            radii=_galactocentric_radius(cut,rotation,offset)
            center_safe=np.isfinite(radii) & (radii >= MIN_CYLINDRICAL_R_KPC)
            removed=int((~center_safe).sum())
            if removed:
                rejected_near_center += removed
                rejected += removed
                LOG.warning('Skipped %d stars with Galactocentric R < %.3f kpc '
                            '(or non-finite R) in rows %d:%d; total skipped=%d',
                            removed,MIN_CYLINDRICAL_R_KPC,start,stop,rejected_near_center)
                cut={f:values[center_safe] for f,values in cut.items()}
            accepted=len(cut['ra'])
            if accepted:
                x,v,c=_convert_chunk(cut,rotation,offset,sun_phi,sun_vel)
                X[count:count+accepted]=x
                V[count:count+accepted]=v
                Cobs[count:count+accepted]=c
                if has_source_id:
                    source_id=rows[column_map['source_id']][start:stop]
                    selected_ids=source_id[good]
                    if removed:
                        selected_ids=selected_ids[center_safe]
                    ids[count:count+accepted]=np.asarray(selected_ids,dtype=np.int64)
                count+=accepted
        if (start//chunk_rows)%10==0 or stop==N:
            LOG.info('Gaia raw preprocessing: %d/%d scanned, %d eligible',stop,N,count)
    if count<100:
        raise ValueError('Fewer than 100 valid Gaia stars after quality cuts')
    geom=int(np.count_nonzero((np.abs(X[:count,0])<2.5) &
                (np.abs(X[:count,1]-8.277)<.25) &
                (np.abs(X[:count,2])<np.deg2rad(3.0))))
    LOG.info('Hapitas-like solar-neighborhood geometry: %d eligible stars',geom)
    if geom==0:
        raise RuntimeError('Zero stars survive geometry after raw Gaia transformation: verify coordinate conventions before model training')
    temp=cache.with_name(cache.stem+'.incomplete.npz')
    try:
        LOG.info('Writing %d selected stars to cache %s',count,cache)
        # Intentionally uncompressed to reduce CPU time for repeated HPC runs.
        arrays=dict(X=X[:count],V=V[:count],Cobs=Cobs[:count])
        if has_source_id:
            arrays['source_id']=ids[:count]
        np.savez(temp,**arrays)
        os.replace(temp,cache)
        description={**expected,'raw_rows':N,'quality_rows':count,
                     'rejected_rows':rejected,'near_galactic_center_rejected':rejected_near_center,
                     'min_galactocentric_radius_kpc':MIN_CYLINDRICAL_R_KPC,
                     'solar_neighborhood_rows':geom,
                     'source_id_present':has_source_id,'resolved_field_names':column_map,
                     'X_columns':['z_kpc','R_kpc','phi_rad_sun_centered'],
                     'V_columns':['vz_km_s','vR_km_s','vphi_km_s_left_handed'],
                     'Cobs':'full 3x3 velocity measurement covariance via first-order 6D Jacobian',
                     'quality':'positive parallax, SNR >= min_snr, RUWE <= max_ruwe, finite astrometry and covariances',
                     'distance_estimator':'1/parallax_mas (not Bayesian)',
                     'solar_parameters':{'R0_kpc':8.277,'z_sun_kpc':0.0208,
                                         'galcen_v_sun_xyz_kms':[9.3,251.5,8.59]},
                     'rv_zero_point_correction':'NOT applied: grvs_mag and rv_template_teff absent from query',
                     'note':'Gaia TAP query, not equivalent to published Hapitas selection in all respects'}
        sidecar.write_text(json.dumps(description,indent=2))
        LOG.warning('Raw query lacks grvs_mag and rv_template_teff: Hapitas RV systematic correction was not applied')
        LOG.info('Gaia cache ready: %s',cache)
        return cache
    finally:
        if temp.exists():temp.unlink()
