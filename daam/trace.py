from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from .utils import cache_dir, auto_autocast
from .experiment import GenerationExperiment
from .heatmap import RawHeatMapCollection, GlobalHeatMap
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator


__all__ = ['trace', 'DiffusionHeatMapHooker', 'GlobalHeatMap']


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            pipeline:
            StableDiffusionPipeline,
            low_memory: bool = False,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: str = None
    ):
        self.all_heat_maps = RawHeatMapCollection()
        self.all_negative_heat_maps = RawHeatMapCollection()
        self.all_uncond_heat_maps = RawHeatMapCollection()
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        self.latent_hw = 4096 if h == 512 else 9216  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        locate_middle = load_heads or save_heads
        self.locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None, locate_middle_block=locate_middle)
        self.last_prompt: str = ''
        self.last_negative_prompt: str = ''
        self.last_image: Image = None
        self.time_idx = 0
        self._gen_idx = 0

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]
        # print(len(modules))

        modules.append(PipelineHooker(pipeline, self))

        super().__init__(modules)
        self.pipe = pipeline
        

    def time_callback(self, *args, **kwargs):
        #print(f'Inference step {self.time_idx}.')
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def to_experiment(self, path, seed=None, id='.', subtype='.', **compute_kwargs):
        # type: (Union[Path, str], int, str, str, Dict[str, Any]) -> GenerationExperiment
        """Exports the last generation call to a serializable generation experiment."""
        #print('000000')
        return GenerationExperiment(
            self.last_image,
            self.compute_global_heat_map(**compute_kwargs).heat_maps,
            self.last_prompt,
            seed=seed,
            id=id,
            subtype=subtype,
            path=path,
            tokenizer=self.pipe.tokenizer,
        )
        

    def compute_activation_ratio(self, prompt=None, negative_prompt=None, bounding_box = None):
        # type: (str,  str, List) -> (List[float], List[float])
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.
            time_idx: Restrict the application to heat maps with this time index. If `None`, use all time steps.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        #print(self.time_idx)
        heat_maps = self.all_heat_maps
        negative_heat_maps = self.all_negative_heat_maps
        #print(negative_heat_maps)
        #print(len(heat_maps.ids_to_heatmaps))
        

        if prompt is None:
            prompt = self.last_prompt
            #print(self.last_prompt)
        if negative_prompt is None:
            negative_prompt = self.last_negative_prompt
            
        # print(len(heat_maps.ids_to_heatmaps))
        # raise ValueError('The shape of maps and negative_maps are not the same.')

        all_merges = [0]*31
        all_merges_negative = [0]*31
        
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (_, _, _, time), heat_map in heat_maps:
                heat_map = heat_map.unsqueeze(1)
                heat_map = F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0)
                if bounding_box is not None:
                    heat_map = heat_map[:,:,bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
                heat_map_value = heat_map.pow(2).mean()
                all_merges[time] += heat_map_value.cpu().numpy()
                
            
            for (_, _, _, time), negative_heat_map in negative_heat_maps:
                negative_heat_map = negative_heat_map.unsqueeze(1)
                negative_heat_map = F.interpolate(negative_heat_map, size=(x, x), mode='bicubic').clamp_(min=0)
                if bounding_box is not None:
                    negative_heat_map = negative_heat_map[:,:,bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
                negative_heat_map_value = negative_heat_map.pow(2).mean()
                all_merges_negative[time] += negative_heat_map_value.cpu().numpy()


        return all_merges, all_merges_negative
    
    
    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False, negative_prompt=None, time_idx=None):
        # type: (str, List[float], List[int], List[int], bool, str, List[int]) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.
            time_idx: Restrict the application to heat maps with this time index. If `None`, use all time steps.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        #print(self.time_idx)
        heat_maps = self.all_heat_maps
        negative_heat_maps = self.all_negative_heat_maps
        #print(negative_heat_maps)
        #print(len(heat_maps.ids_to_heatmaps))
        

        if prompt is None:
            prompt = self.last_prompt
            #print(self.last_prompt)
        if negative_prompt is None:
            negative_prompt = self.last_negative_prompt

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)
            
        # print(len(heat_maps.ids_to_heatmaps))
        # raise ValueError('The shape of maps and negative_maps are not the same.')

        all_merges = []
        all_merges_negative = []
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head, time), heat_map in heat_maps:
                #print(type(heat_maps))
                #print(factor, layer, head, time, 'heat_map',heat_map)
                if factor in factors and (head_idx is None or head in head_idx) and (layer_idx is None or layer in layer_idx) and (time_idx is None or time in time_idx):
                    #print(heat_map.shape)
                    heat_map = heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))
            
            for (factor, layer, head, time), negative_heat_map in negative_heat_maps:
                if factor in factors and (head_idx is None or head in head_idx) and (layer_idx is None or layer in layer_idx) and (time_idx is None or time in time_idx):
                    #print(heat_map.shape)
                    negative_heat_map = negative_heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges_negative.append(F.interpolate(negative_heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0)
                negative_maps = torch.stack(all_merges_negative, dim=0)
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            # TODO: fix this
            #print(maps.shape)
            maps = maps.mean(0)[:, 0]
            negative_maps = negative_maps.mean(0)[:, 0]
            
            maps = maps[:len(self.pipe.tokenizer.tokenize(prompt)) + 2]  # 1 for SOS and 1 for padding
            negative_maps = negative_maps[:len(self.pipe.tokenizer.tokenize(negative_prompt)) + 2]
            #print(maps.shape)

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities
                negative_maps = negative_maps / (negative_maps[1:-1].sum(0, keepdim=True) + 1e-6)
            #print(maps.shape)
        return GlobalHeatMap(self.pipe.tokenizer, prompt, maps, negative_prompt, negative_maps)
    
    def return_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False, negative_prompt=None, time_idx=None):
        # type: (str, List[float], List[int], List[int], bool, str, List[int]) -> GlobalHeatMap
        
        negative_heat_maps = self.all_negative_heat_maps
        
        if negative_prompt is None:
            negative_prompt = self.last_negative_prompt

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)
            
        # print(len(heat_maps.ids_to_heatmaps))
        # raise ValueError('The shape of maps and negative_maps are not the same.')

        all_merges_negative = [[] for i in range(31)]
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):            
            for (factor, layer, head, time), negative_heat_map in negative_heat_maps:
                if factor in factors and (head_idx is None or head in head_idx) and (layer_idx is None or layer in layer_idx):
                    # print(factor, layer, head, time, negative_heat_map.shape)
                    #print(heat_map.shape)
                    negative_heat_map = negative_heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges_negative[time].append(F.interpolate(negative_heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                
                negative_maps = [torch.stack(all_merges_negative_, dim=0) for all_merges_negative_ in all_merges_negative]
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            # TODO: fix this
            #print(maps.shape)
            #print(negative_maps[0].shape)
            negative_maps = [negative_map.mean(0)[:, 0] for negative_map in negative_maps]
            #print(negative_maps[0].shape)
            
            for i in range(len(negative_maps)):
                negative_maps[i] = negative_maps[i][1:len(self.pipe.tokenizer.tokenize(negative_prompt)) + 1] # 1 for SOS and 1 for padding
                if normalize:
                    negative_maps[i] = negative_maps[i] / (negative_maps[i][1:-1].sum(0, keepdim=True) + 1e-6)
                negative_maps[i] = negative_maps[i].mean(0,keepdim=True)
            #print(maps.shape)
            
        return negative_maps
        


class PipelineHooker(ObjectHooker[StableDiffusionPipeline]):
    def __init__(self, pipeline: StableDiffusionPipeline, parent_trace: 'trace'):
        super().__init__(pipeline)
        self.heat_maps = parent_trace.all_heat_maps
        self.negative_heat_maps = parent_trace.all_negative_heat_maps
        self.uncond_heat_maps = parent_trace.all_uncond_heat_maps
        self.parent_trace = parent_trace

    def _hooked_run_safety_checker(hk_self, self: StableDiffusionPipeline, image, *args, **kwargs):
        image, has_nsfw = hk_self.monkey_super('run_safety_checker', image, *args, **kwargs)
        pil_image = self.numpy_to_pil(image)
        hk_self.parent_trace.last_image = pil_image[0]

        return image, has_nsfw

    def _hooked_encode_prompt(hk_self, _: StableDiffusionPipeline, prompt: Union[str, List[str]], *args, **kwargs):
        #print(prompt)
        # TODO: fix this 
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt
            
        # TODO: fix this 
        #print(args[-1])
        if args[-1] is not None:
            print(args[-1])
            if not isinstance(args[-1], str) and len(args[-1]) > 1:
                raise ValueError('Only single prompt generation is supported for heat map computation.')
            elif not isinstance(args[-1], str):
                last_negative_prompt = args[-1][0]
            else:
                last_negative_prompt = args[-1]
        else:
            last_negative_prompt = ''
                
                
        #print(last_prompt)
        hk_self.heat_maps.clear()
        hk_self.negative_heat_maps.clear()
        hk_self.uncond_heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt
        hk_self.parent_trace.last_negative_prompt = last_negative_prompt
        ret = hk_self.monkey_super('_encode_prompt', prompt, *args, **kwargs)
        #print(ret.shape)
        return ret
    
    def _hooked_encode_prompt_total(hk_self, _: StableDiffusionPipeline, prompt: Union[str, List[str]], *args, **kwargs):
        #print(prompt)
        # TODO: fix this 
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt
            
        # TODO: fix this 
        print(args)
        if args[-1] is not None:
            print(args[-1])
            if not isinstance(args[-1], str) and len(args[-1]) > 1:
                raise ValueError('Only single prompt generation is supported for heat map computation.')
            elif not isinstance(args[-1], str):
                last_negative_prompt = args[-1][0]
            else:
                last_negative_prompt = args[-1]
        else:
            last_negative_prompt = ''
                
                
        #print(last_prompt)
        hk_self.heat_maps.clear()
        hk_self.negative_heat_maps.clear()
        hk_self.uncond_heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt
        hk_self.parent_trace.last_negative_prompt = last_negative_prompt
        ret = hk_self.monkey_super('_encode_prompt', prompt, *args, **kwargs)
        #print(ret.shape)
        return ret
    

    def _hook_impl(self):
        self.monkey_patch('run_safety_checker', self._hooked_run_safety_checker)
        self.monkey_patch('_encode_prompt', self._hooked_encode_prompt)
        self.monkey_patch('_encode_prompt_total', self._hooked_encode_prompt_total)
        


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
            self,
            module: Attention,
            parent_trace: 'trace',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: Union[str, Path] = None,
    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.negative_heat_maps = parent_trace.all_negative_heat_maps
        self.uncond_heat_maps = parent_trace.all_uncond_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / 'heads'

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        negative_maps = []
        #print(x.shape)
        x = x.permute(2, 0, 1)
        #print(x.shape)
        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                #print(map_.shape)
                map_ = map_.view(map_.size(0), h, w)
                
                if map_.size(0) % 3 == 0:
                    pos_map_ = map_[map_.size(0) // 3:map_.size(0) // 3 *2]
                    negative_map_ = map_[:map_.size(0) // 3]
                else:
                    # original code
                    pos_map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                    negative_map_ = map_[:map_.size(0) // 2]
                #print(map_.shape)
                maps.append(pos_map_)
                negative_maps.append(negative_map_)
                #import sys
                #sys.exit(1)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        negative_maps = torch.stack(negative_maps, 0)
        #print(maps.shape)
        return maps.permute(1, 0, 2, 3).contiguous(), negative_maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f'{self.trace._gen_idx}.pt')

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f'{self.trace._gen_idx}.pt')

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        #encoder_hidden_states  = None
        print(encoder_hidden_states.shape if encoder_hidden_states is not None else None)
        print(hidden_states.shape)
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #print('hidden_states',hidden_states.shape)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
 
        query = attn.head_to_batch_dim(query)
        #print(query.shape)
        key = attn.head_to_batch_dim(key)
        #print('key',key.shape)
        value = attn.head_to_batch_dim(value)
        #print(value.shape)
        raise ValueError('The shape of maps and negative_maps are not the same.')
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        #print(attention_probs.shape)
        #print(attention_probs.shape)


        # DAAM save heads
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            attention_probs = self._load_attn()

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
        #print(self.trace._gen_idx//15)
        # print(attention_probs.shape)

        # skip if too large
        if attention_probs.shape[-1] == self.context_size and factor != 8:
            
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            #print(attention_probs.shape)
            maps, negative_maps = self._unravel_attn(attention_probs)
            if maps.shape != negative_maps.shape:
                raise ValueError('The shape of maps and negative_maps are not the same.')
            # print(maps.shape)
            #print('factor')
            
            # TODO
            for head_idx, heatmap in enumerate(maps):
                #print('hit')
                self.heat_maps.update(factor, self.layer_idx, head_idx, self.trace._gen_idx//15, heatmap)
            for head_idx, negative_heatmap in enumerate(negative_maps):
                self.negative_heat_maps.update(factor, self.layer_idx, head_idx, self.trace._gen_idx//15, negative_heatmap)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # print(hidden_states.shape)
        # import sys
        # sys.exit(1)
        
        self.trace._gen_idx += 1

        return hidden_states

    def _hook_impl(self):
        self.module.set_processor(self)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
