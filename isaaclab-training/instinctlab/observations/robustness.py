"""R6 range-frame impairment before the shared actor/ICS history interface.

Dormant until explicitly constructed by a registered R6 workload. This module
accepts only measured ranges, validity and time. Clean copies are audit output,
never inputs to the actor/control history. It does not alter the actor schema.
"""
import hashlib
import math
import torch


class RangeDelivery:
    def __init__(self, *, lanes, rays, seed, stream_id, sigma_m=0., delay_cycles=0,
                 device='cpu', dtype=torch.float32):
        if type(lanes) is not int or lanes<=0 or type(rays) is not int or rays<=0:
            raise ValueError('positive tensor dimensions required')
        if type(seed) is not int or not isinstance(stream_id,str) or not stream_id:
            raise ValueError('explicit random stream identity required')
        if type(sigma_m) not in (int,float) or not math.isfinite(sigma_m) or not 0<=sigma_m<=.03:
            raise ValueError('unregistered noise range')
        if type(delay_cycles) is not int or delay_cycles not in (0,1):
            raise ValueError('unregistered delay')
        if dtype not in (torch.float32,torch.float64): raise ValueError('floating dtype required')
        self.lanes,self.rays=lanes,rays
        self.sigma,self.delay=float(sigma_m),delay_cycles
        self.device,self.dtype=torch.device(device),dtype
        self.generators=[];self.streams=[]
        for lane in range(lanes):
            # A single CPU algorithm is replayable without CUDA and independent
            # of the device running physics. This is a new stream identity.
            payload=f'R6/range-noise/cpu-v1/{seed}/{stream_id}/{lane}'.encode()
            digest=hashlib.sha256(payload).hexdigest()
            generator=torch.Generator(device='cpu')
            generator.manual_seed(int(digest[:16],16)%(2**63))
            self.generators.append(generator);self.streams.append(digest)
        self.pending=self._empty();self.last=self._empty()
        self.last_time=torch.full((lanes,1),-1.,device=self.device,dtype=dtype)

    def _empty(self):
        return dict(ranges=torch.full((self.lanes,self.rays),40.,device=self.device,dtype=self.dtype),
                    valid=torch.zeros((self.lanes,self.rays),device=self.device,dtype=torch.bool),
                    time=torch.zeros((self.lanes,1),device=self.device,dtype=self.dtype),
                    available=torch.zeros((self.lanes,1),device=self.device,dtype=torch.bool))

    def reset(self, indices):
        if not isinstance(indices,torch.Tensor) or indices.dtype!=torch.long or indices.ndim!=1 or indices.device!=self.device:
            raise ValueError('reset indices must be device long vector')
        if bool(((indices<0)|(indices>=self.lanes)).any()): raise ValueError('reset index range')
        for frame in (self.pending,self.last):
            frame['ranges'][indices]=40.;frame['valid'][indices]=False
            frame['time'][indices]=0.;frame['available'][indices]=False
        self.last_time[indices]=-1.
        # Noise streams continue across physical resets; reset does not silently
        # replay the same noise. Full RNG states belong in the experiment record.

    def rng_states(self):
        return [g.get_state().clone() for g in self.generators]

    def deliver(self, ranges_m, validity, acquisition_time_s, drop):
        shape=(self.lanes,self.rays)
        for value,expected,dtype in ((ranges_m,shape,self.dtype),(validity,shape,torch.bool),
                                     (acquisition_time_s,(self.lanes,1),self.dtype),
                                     (drop,(self.lanes,1),torch.bool)):
            if not isinstance(value,torch.Tensor) or value.shape!=expected or value.dtype!=dtype or value.device!=self.device:
                raise ValueError('measurement shape/device/dtype mismatch')
        if not bool(torch.isfinite(ranges_m).all()) or not bool(torch.isfinite(acquisition_time_s).all()):
            raise ValueError('nonfinite producer input')
        if bool((acquisition_time_s<0).any()) or bool((acquisition_time_s<=self.last_time).any()):
            raise ValueError('acquisition timestamp must advance; reset lane explicitly')
        if bool(((ranges_m<.1)|(ranges_m>40.)).any()):
            raise ValueError('producer range outside approved bounds')
        if bool(((~validity)&(ranges_m!=40.)).any()):
            raise ValueError('invalid beam must retain exact 40m fill')
        noisy=ranges_m.clone()
        if self.sigma:
            noisy=ranges_m.detach().cpu().clone()
            cpu_validity=validity.detach().cpu()
            for lane,generator in enumerate(self.generators):
                mask=cpu_validity[lane]
                # Draw by acquisition/lane/beam, never by a policy-dependent
                # validity mask. Paired policies therefore receive the same
                # noise on shared beams and keep identical subsequent streams.
                noise=torch.randn((self.rays,),device='cpu',dtype=self.dtype,generator=generator)
                noisy[lane,mask]=(noisy[lane,mask]+noise[mask].clamp(-3.,3.)*self.sigma).clamp(.1,40.)
            noisy=noisy.to(self.device)
        acquired=dict(ranges=noisy,valid=validity.clone(),time=acquisition_time_s.clone(),
                      available=torch.ones((self.lanes,1),device=self.device,dtype=torch.bool))
        candidate=acquired if self.delay==0 else self.pending
        delivered={key:torch.where(drop,self.last[key],value).clone() for key,value in candidate.items()}
        self.pending={k:v.clone() for k,v in acquired.items()}
        self.last={k:v.clone() for k,v in delivered.items()}
        self.last_time.copy_(acquisition_time_s)
        return dict(clean_ranges=ranges_m.clone(),clean_validity=validity.clone(),
                    delivered_ranges=delivered['ranges'],delivered_validity=delivered['valid'],
                    acquisition_time=delivered['time'],delivery_time=acquisition_time_s.clone(),
                    frame_age=acquisition_time_s-delivered['time'],drop=drop.clone(),
                    frame_available=delivered['available'],noise_rng_identity=list(self.streams))
