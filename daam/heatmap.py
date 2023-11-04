from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable

from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import spacy.tokens
import torch
import torch.nn.functional as F

from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 24})
        plt_ = plt
    else:
        plt_ = ax

    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
        else:
            heat_map = heat_map.clamp_(min=0, max=1)
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        plt_.imshow(im)

        if word is not None:
            if ax is None:
                plt.title(word)
            else:
                ax.set_title(word)

        if out_file is not None:
            plt.savefig(out_file)



def plot_overlay_heat_map_with_raw(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes | None) -> None
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 16})
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create 1 row, 2 columns of subplots
    else:
        # If ax is provided, we assume it is a list or tuple of Axes objects
        if not isinstance(ax, (list, tuple)) or len(ax) != 2:
            raise ValueError("ax must be a list or tuple of two matplotlib Axes if provided.")
        axs = ax

    # Remove x and y axis for all subplots
    for ax in axs:
        ax.axis('off')

    # Original Image
    axs[0].imshow(np.array(im))

    with torch.no_grad():  # Wrap in no_grad context if not using auto_autocast
        # Heatmap overlay processing
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            heatmap_normalized = heat_map.squeeze().cpu().numpy()
        else:
            heatmap_normalized = heat_map.clamp_(min=0, max=1).squeeze().cpu().numpy()

        # Display heatmap
        axs[1].imshow(heatmap_normalized, cmap='jet', alpha=0.5)  # Set alpha for transparency

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)

        # Display image with heatmap overlay
        axs[1].imshow(im.numpy())

    # Titles
    if word is not None:
        axs[0].set_title(f"Original Image: {word}")
        axs[1].set_title(f"Heatmap Overlay: {word}")

    # If out_file is provided, save the figure to the file
    if out_file is not None:
        print(out_file)
        plt.savefig(out_file, bbox_inches='tight')
    plt.show()  # Display the figure



class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(
            image,
            self.expand_as(image, **expand_kwargs),
            word=self.word,
            out_file=out_file,
            color_normalize=color_normalize,
            ax=ax
        )
    def plot_overlay_with_raw(self, image, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map_with_raw(
            image,
            self.expand_as(image, **expand_kwargs),
            word=self.word,
            out_file=out_file,
            color_normalize=color_normalize,
            ax=ax
        )
        
    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor
        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor, negative_prompt: str, negative_heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.negative_heat_maps = negative_heat_maps
        self.negative_prompt = negative_prompt
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        #print(word,word_idx,offset_idx)
        negative_merge_idxs, negative_word_idx = compute_token_merge_indices(self.tokenizer, self.negative_prompt, word, word_idx, offset_idx)
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        if negative_merge_idxs:
            #print("negative")
            return WordHeatMap(self.negative_heat_maps[negative_merge_idxs].mean(0), word, word_idx)
        elif merge_idxs:
            #print('positive')
            return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)
        else:
            raise ValueError(f'Word {word} not found in prompt')

    def parsed_heat_maps(self) -> Iterable[ParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            try:
                heat_map = self.compute_word_heat_map(token.text)
                yield ParsedHeatMap(heat_map, token)
            except ValueError:
                pass

    def dependency_relations(self) -> Iterable[SyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != 'ROOT':
                try:
                    dep_heat_map = self.compute_word_heat_map(token.text)
                    head_heat_map = self.compute_word_heat_map(token.head.text)

                    yield SyntacticHeatMapPair(head_heat_map, dep_heat_map, token.head.text, token.text, token.dep_)
                except ValueError:
                    pass


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
