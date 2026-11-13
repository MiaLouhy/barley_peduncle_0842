"""
New step-by-step barley CT analysis workflow.

The first step implemented here removes the holder tube from a REK volume while
keeping the API close to the old analysis code.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_otsu
from skimage.measure import CircleModel, ransac
from skimage.morphology import remove_small_objects

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import nrrd
except ImportError:
    nrrd = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class Tube:
    """Volume wrapper used by the new step-by-step workflow."""

    def __init__(self, data_path, source_type=None):
        self.filename = str(data_path)
        self.source_type = source_type
        self.data = self._read_input_data(data_path, source_type=source_type)
        self.segmented_data = self.data
        self.tube_removed_data = None
        self.otsu_filtered_data = None
        self.peduncle_data = None
        self.peduncle_structure_data = None
        self.lumen_data = None
        self.pseudo_label_data = None
        self.pseudo_label_result = None
        self.wall_thickness_result = None
        self.cross_section_metrics_result = None
        self.pixel_intensity_result = None
        self.peduncle_bounds = None

    @staticmethod
    def _read_input_data(data_path, source_type=None):
        if data_path is None:
            raise ValueError("data_path cannot be None")

        if isinstance(data_path, np.ndarray):
            return data_path

        path = Path(data_path)
        suffix = path.suffix.lower()

        if source_type is None:
            if suffix in {".rek"}:
                source_type = "rek"
            elif suffix in {".tif", ".tiff"}:
                source_type = "tiff"
            else:
                raise ValueError(f"Unsupported file type: {path}")

        if source_type == "rek":
            return Tube._read_rek_file(path)

        if source_type == "tiff":
            return Tube._read_tiff_file(path)

        raise ValueError(f"Unsupported source_type: {source_type}")

    @staticmethod
    def _read_rek_file(rek_path):
        with open(rek_path, mode="rb") as file:
            hdr_bytes = file.read(2 * 1024)
            hdr = np.frombuffer(hdr_bytes, dtype=np.uint16)
            shape = (hdr[3], hdr[1], hdr[0])

        return np.memmap(
            rek_path,
            offset=2048,
            dtype="uint16",
            shape=shape,
            mode="r",
        )

    @staticmethod
    def _read_tiff_file(tiff_path):
        if tifffile is None:
            raise ImportError("tifffile is required to read TIFF volumes")

        data = np.asarray(tifffile.imread(str(tiff_path)))
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D TIFF data, got shape {data.shape}")
        return data

    @staticmethod
    def _read_nrrd_file(nrrd_path):
        if nrrd is not None:
            data, _ = nrrd.read(str(nrrd_path))
            return np.asarray(data)

        if sitk is not None:
            image = sitk.ReadImage(str(nrrd_path))
            return np.asarray(sitk.GetArrayFromImage(image))

        try:
            from skimage import io

            return np.asarray(io.imread(str(nrrd_path)))
        except Exception as exc:
            raise ImportError(
                "Reading .nrrd files requires the 'pynrrd' or 'SimpleITK' package."
            ) from exc

    def load_segmented_data_from_nrrd(self, file_path):
        """Read a single .nrrd segmentation mask and align it with the raw volume."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"NRRD file not found: {path}")

        if path.suffix.lower() != ".nrrd":
            raise ValueError(f"Expected a .nrrd file, got: {path}")

        arr = self._read_nrrd_file(path)
        arr = np.asarray(arr)

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0][np.newaxis, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"Unsupported NRRD array shape {arr.shape} for {path}")

        if self.data is not None and self.data.ndim == 3:
            target_shape = self.data.shape
            if arr.shape == target_shape:
                aligned = arr
            elif arr.shape == target_shape[::-1]:
                aligned = np.transpose(arr, (2, 1, 0))
            else:
                raise ValueError(
                    f"Cannot align NRRD shape {arr.shape} with volume shape {target_shape}"
                )
        else:
            aligned = arr

        mask = (aligned > 0).astype(bool)
        if not np.any(mask):
            raise ValueError(f"NRRD segmentation mask is empty: {path}")

        if self.data.shape != mask.shape:
            raise ValueError(
                f"NRRD mask shape {mask.shape} does not match raw volume shape {self.data.shape}"
            )

        masked_data = np.where(mask, self.data, 0)
        self.segmented_data = masked_data.astype(self.data.dtype)


    def export_segmented_tiff(self, output_path, data=None, overwrite=False, compress=False):
        """Export the current segmented data to a TIFF file."""
        if tifffile is None:
            raise ImportError("tifffile is required to export TIFF volumes")

        target = self.segmented_data if data is None else data
        if target is None:
            raise ValueError("No segmented data available to export")

        target = np.asarray(target)
        if target.ndim not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D data, got shape {target.shape}")

        output_path = Path(output_path)
        if output_path.suffix.lower() not in {".tif", ".tiff"}:
            output_path = output_path.with_suffix(".tif")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            print(f"Skip export: target TIFF already exists -> {output_path}")
            return output_path

        if compress:
            target = crop_nonzero_data(target)

        tifffile.imwrite(
            str(output_path),
            target,
            metadata={"axes": "ZYX"} if target.ndim == 3 else {},
            compression="zlib" if compress else None,
        )
        return output_path

    def remove_holder_tube(
        self,
        start_slice=500,
        stop_slice=None,
        tube_r=160,
        tube_thickness=30,
        sample_step=100,
        outlier_px=12,
        show_progress=False,
        debug=False,
        method="angular",
        radial_margin=2,
        n_angles=360,
        background_cutoff=None,
        safe_removal=0,
    ):
        """Remove holder while preserving source REK coordinates (Z,Y,X).

        method="angular": adaptation of PhenoCT's newer
        Tube.segment_sample_holder(). Uses a maximum-intensity projection to
        estimate an angle-dependent holder inner boundary. Unlike upstream,
        this method does not crop Z, and by default does not threshold tissue;
        the caller's existing cleanup step can do that afterwards.

        method="legacy": retain the prior RANSAC circle + fixed inner radius.

        Stop slice is exclusive. Outside [start_slice:stop_slice], output is zero.
        ``background_cutoff`` optionally enables PhenoCT's histogram-based
        background threshold (e.g. 2.0); default None avoids applying a second
        threshold before the existing denoiser. ``safe_removal`` optionally
        removes small 3D components, which is also disabled by default.
        """
        if method == "angular":
            self.tube_removed_data = remove_holder_tube_angular(
                self.data,
                start_slice=start_slice,
                stop_slice=stop_slice,
                radial_margin=radial_margin,
                n_angles=n_angles,
                show_progress=show_progress,
                debug=debug,
                background_cutoff=background_cutoff,
                safe_removal=safe_removal,
            )
        elif method == "legacy":
            self.tube_removed_data = remove_holder_tube(
                self.data,
                start_slice=start_slice,
                stop_slice=stop_slice,
                tube_r=tube_r,
                tube_thickness=tube_thickness,
                sample_step=sample_step,
                outlier_px=outlier_px,
                show_progress=show_progress,
            )
        else:
            raise ValueError("method must be 'angular' or 'legacy'")

        self.segmented_data = self.tube_removed_data
        if debug:
            self.show_data_and_segmented()

    def filter_by_otsu(
        self,
        otsu_factor=0.4,
        verbose=False,
        debug=False,
    ):
        """
        Filter tube_removed_data with one global relaxed Otsu threshold.

        The Otsu threshold is computed from all non-zero voxels in the current
        tube_removed_data. Voxels above threshold_otsu * otsu_factor keep their
        original intensity.
        """
        source = self.tube_removed_data if self.tube_removed_data is not None else self.segmented_data
        self.otsu_filtered_data = filter_by_otsu(
            source,
            otsu_factor=otsu_factor,
            verbose=verbose or debug,
        )
        self.segmented_data = self.otsu_filtered_data
        if debug:
            self.show_data_and_segmented()

    def threshold_denoise_and_remove_islands(
        self,
        noise_threshold=1500,
        min_component_size=3000,
        source=None,
        verbose=False,
        debug=False,
    ):
        """
        Remove low-intensity background, then remove small islands.

        Voxels below noise_threshold are zeroed first. Connected components in
        the remaining mask smaller than min_component_size are removed, matching
        the 3D Slicer Islands-style minimum component size cleanup.
        """
        if source is None:
            source = self.segmented_data if self.segmented_data is not None else self.data

        cleaned_data, stats = threshold_denoise_and_remove_islands(
            source,
            noise_threshold=noise_threshold,
            min_component_size=min_component_size,
        )
        self.segmented_data = cleaned_data

        if verbose or debug:
            print(
                "Threshold denoise/island cleanup: "
                f"noise_threshold>={noise_threshold}, "
                f"min_component_size>={min_component_size}, "
                f"components {stats['before_components']} -> {stats['after_components']}, "
                f"voxels {stats['before_voxels']} -> {stats['after_voxels']}"
            )

        if debug:
            self.show_data_and_segmented()

        return stats

    def process_rek_and_calculate_structure_metrics(
        self,
        remove_holder_kwargs=None,
        denoise_kwargs=None,
        metric_kwargs=None,
        debug=False,
    ):
        """Run the standard REK cleanup pipeline and return cross-section metrics.

        The default sequence contains: remove the holder tube, threshold and remove 
        small islands, then calculate metrics for the main peduncle structure. 
        Use the optional keyword dictionaries to tune the corresponding processing step.
        """
        remove_holder_kwargs = (
            {} if remove_holder_kwargs is None else dict(remove_holder_kwargs)
        )
        denoise_kwargs = {} if denoise_kwargs is None else dict(denoise_kwargs)
        metric_kwargs = {} if metric_kwargs is None else dict(metric_kwargs)

        self.remove_holder_tube(**remove_holder_kwargs)
        self.threshold_denoise_and_remove_islands(**denoise_kwargs)
        return self.calculate_cross_section_metrics(
            debug=debug,
            **metric_kwargs,
        )

    def locate_peduncle(
        self,
        smooth_win=21,
        slice_mm=0.084,
        max_len_mm=50.0,
        head_margin_slices=12,
        tail_margin_slices=12,
        rise_threshold=200,
        verbose=False,
        debug=False,
    ):
        """
        Locate the peduncle region and keep only that z-range.

        This uses the legacy area-gradient approach: find the spike-end drop,
        find a stable low-area peduncle region, then find the stem-start rise.
        """
        source = self.otsu_filtered_data if self.otsu_filtered_data is not None else self.segmented_data
        z1, z2 = detect_peduncle_boundaries(
            source,
            smooth_win=smooth_win,
            slice_mm=slice_mm,
            max_len_mm=max_len_mm,
            head_margin_slices=head_margin_slices,
            tail_margin_slices=tail_margin_slices,
            rise_threshold=rise_threshold,
            verbose=verbose or debug,
        )
        self.peduncle_bounds = (z1, z2)

        self.peduncle_data = np.zeros_like(source)
        self.peduncle_data[z1:z2] = source[z1:z2]
        self.segmented_data = self.peduncle_data

        if debug:
            plot_peduncle_area_profile(source, z1, z2)

    def calculate_wall_thickness(
        self,
        px_xy_mm=0.084,
        min_component_size=3000,
        closing_radius=1,
        verbose=False,
        debug=False,
    ):
        """
        Extract the main 3D peduncle structure and calculate wall thickness.

        The main structure is extracted from self.peduncle_data. Thickness is
        calculated per slice from equivalent outer and lumen radii.
        """
        source = self.peduncle_data if self.peduncle_data is not None else self.segmented_data
        structure_mask, structure_stats = extract_main_3d_structure(
            source > 0,
            min_component_size=min_component_size,
        )
        self.peduncle_structure_data = np.where(structure_mask, source, 0).astype(source.dtype)
        self.segmented_data = self.peduncle_structure_data

        self.wall_thickness_result = calculate_wall_thickness_from_mask(
            structure_mask,
            px_xy_mm=px_xy_mm,
            closing_radius=closing_radius,
            verbose=verbose or debug,
        )
        self.wall_thickness_result["structure_stats"] = structure_stats
        lumen_mask = self.wall_thickness_result["lumen_mask"]
        self.lumen_data = (lumen_mask.astype(np.uint8) * 255)

        if verbose or debug:
            print(
                "Main 3D structure: "
                f"components {structure_stats['before_components']} -> "
                f"{structure_stats['after_small_components']} -> 1, "
                f"voxels {structure_stats['before_voxels']} -> "
                f"{structure_stats['after_main_voxels']}, "
                f"active_slices {structure_stats['before_active_slices']} -> "
                f"{structure_stats['after_active_slices']}"
            )

        if debug:
            plot_wall_thickness_profile(self.wall_thickness_result)
            show_wall_thickness_debug(
                self.data,
                self.peduncle_structure_data,
                self.lumen_data,
            )

        return self.wall_thickness_result

    def calculate_cross_section_metrics(
        self,
        px_xy_mm=0.084,
        min_component_size=3000,
        closing_radius=1,
        lumen_quantile=None,
        min_lumen_area_px=50,
        auto_locate_peduncle=True,
        smooth_win=21,
        max_len_mm=50.0,
        head_margin_slices=12,
        tail_margin_slices=12,
        rise_threshold=200,
        slice_mm=None,
        verbose=False,
        debug=False,
    ):
        """
        Calculate summary cross-section metrics for the main peduncle structure.

        This is the current primary metric path. If a main peduncle structure
        has not already been prepared, this method can locate the peduncle
        z-range, extract the largest cleaned 3D structure, and then calculate
        the cross-section metrics directly.
        """
        if self.peduncle_structure_data is not None:
            source = self.peduncle_structure_data
            structure_mask = source > 0
            structure_stats = None
        else:
            if slice_mm is None:
                slice_mm = px_xy_mm

            source = self.peduncle_data if self.peduncle_data is not None else self.segmented_data
            if source is None:
                raise ValueError("No structure data available for cross-section metrics")

            if auto_locate_peduncle and self.peduncle_data is None:
                z1, z2 = detect_peduncle_boundaries(
                    source,
                    smooth_win=smooth_win,
                    slice_mm=slice_mm,
                    max_len_mm=max_len_mm,
                    head_margin_slices=head_margin_slices,
                    tail_margin_slices=tail_margin_slices,
                    rise_threshold=rise_threshold,
                    verbose=verbose or debug,
                )
                self.peduncle_bounds = (z1, z2)
                self.peduncle_data = np.zeros_like(source)
                self.peduncle_data[z1:z2] = source[z1:z2]
                source = self.peduncle_data

                if debug:
                    plot_peduncle_area_profile(self.segmented_data, z1, z2)

            structure_mask, structure_stats = extract_main_3d_structure(
                source > 0,
                min_component_size=min_component_size,
            )
            self.peduncle_structure_data = np.where(structure_mask, source, 0).astype(source.dtype)
            self.segmented_data = self.peduncle_structure_data
            source = self.peduncle_structure_data

        if structure_mask is None:
            raise ValueError("No structure data available for cross-section metrics")

        self.cross_section_metrics_result = calculate_cross_section_metrics_from_mask(
            structure_mask,
            intensity_data=source,
            px_xy_mm=px_xy_mm,
            closing_radius=closing_radius,
            lumen_quantile=lumen_quantile,
            min_lumen_area_px=min_lumen_area_px,
            verbose=verbose or debug,
        )

        if verbose or debug:
            if structure_stats is not None:
                print(
                    "Main 3D structure: "
                    f"components {structure_stats['before_components']} -> "
                    f"{structure_stats['after_small_components']} -> 1, "
                    f"voxels {structure_stats['before_voxels']} -> "
                    f"{structure_stats['after_main_voxels']}, "
                    f"active_slices {structure_stats['before_active_slices']} -> "
                    f"{structure_stats['after_active_slices']}"
                )
            print(
                "Cross-section summary: "
                f"valid_slices={self.cross_section_metrics_result['valid_slices']}, "
                f"d_out={self.cross_section_metrics_result['d_out_mm']:.3f} mm, "
                f"d_in={self.cross_section_metrics_result['d_in_mm']:.3f} mm, "
                f"wt={self.cross_section_metrics_result['wt_mm']:.3f} mm"
            )

        return self.cross_section_metrics_result

    def calculate_pixel_intensity(
        self,
        verbose=False,
        debug=False,
    ):
        """Calculate per-slice pixel intensity from the final main structure."""
        source = (
            self.peduncle_structure_data
            if self.peduncle_structure_data is not None
            else self.segmented_data
        )
        structure_mask = source > 0
        active_z = np.flatnonzero(np.any(structure_mask, axis=(1, 2)))

        area_px = []
        slice_sum_intensity = []
        slice_mean_intensity = []

        for z in active_z:
            mask = structure_mask[z]
            values = source[z][mask].astype(np.float64)
            area_px.append(int(mask.sum()))
            slice_sum_intensity.append(float(values.sum()))
            slice_mean_intensity.append(float(values.mean()) if values.size else 0.0)

        area_px = np.array(area_px, dtype=int)
        slice_sum_intensity = np.array(slice_sum_intensity, dtype=float)
        slice_mean_intensity = np.array(slice_mean_intensity, dtype=float)

        self.pixel_intensity_result = {
            "kept_z": active_z.astype(int),
            "source": "peduncle_structure_data"
            if self.peduncle_structure_data is not None
            else "segmented_data",
            "area_px": area_px,
            "slice_sum_intensity": slice_sum_intensity,
            "slice_mean_intensity": slice_mean_intensity,
            "mean_pixel_intensity": float(slice_mean_intensity.mean())
            if slice_mean_intensity.size
            else 0.0,
            "median_pixel_intensity": float(np.median(slice_mean_intensity))
            if slice_mean_intensity.size
            else 0.0,
            "mean_slice_sum_intensity": float(slice_sum_intensity.mean())
            if slice_sum_intensity.size
            else 0.0,
            "median_slice_sum_intensity": float(np.median(slice_sum_intensity))
            if slice_sum_intensity.size
            else 0.0,
        }

        if verbose or debug:
            print(
                "Pixel intensity: "
                f"kept_slices={active_z.size}, "
                f"mean_pixel={self.pixel_intensity_result['mean_pixel_intensity']:.2f}, "
                f"median_pixel={self.pixel_intensity_result['median_pixel_intensity']:.2f}, "
                f"mean_slice_sum={self.pixel_intensity_result['mean_slice_sum_intensity']:.2f}"
            )

        if debug:
            plot_pixel_intensity_profile(self.pixel_intensity_result)

        return self.pixel_intensity_result

    def generate_pseudo_label(
        self,
        threshold=1500,
        min_component_size=1000,
        source=None,
        update_segmented=True,
        verbose=False,
    ):
        """
        Generate a pseudo label from high-intensity voxels.

        Voxels with intensity >= threshold are kept first, then 3D connected
        components smaller than min_component_size voxels are removed. The
        binary mask is stored in self.pseudo_label_data. By default, the masked
        original intensity data is stored in self.segmented_data.
        """
        if source is None:
            source = self.data

        mask, stats = generate_pseudo_label(
            source,
            threshold=threshold,
            min_component_size=min_component_size,
        )

        self.pseudo_label_data = mask.astype(np.uint8)
        self.pseudo_label_result = stats

        if update_segmented:
            self.segmented_data = np.where(mask, source, 0).astype(source.dtype)

        if verbose:
            print(
                "Pseudo label: "
                f"threshold>={threshold}, "
                f"components {stats['before_components']} -> {stats['after_components']}, "
                f"voxels {stats['before_voxels']} -> {stats['after_voxels']}"
            )


    def plot_raw_intensity_histogram(
        self,
        bins=256,
        exclude_zero=True,
        intensity_range=None,
        log_y=False,
        title=None,
    ):
        """Plot the grayscale intensity histogram of the original raw data."""
        if title is None:
            title = f"Raw data intensity histogram: {Path(self.filename).name}"

        return plot_intensity_histogram(
            self.data,
            bins=bins,
            exclude_zero=exclude_zero,
            intensity_range=intensity_range,
            log_y=log_y,
            title=title,
        )

    def show_data_and_segmented(
        self,
        raw_colormap="cyan",
        segmented_colormap="yellow",
        removed_colormap="red",
        raw_opacity=0.35,
        segmented_opacity=0.65,
        removed_opacity=0.5,
        show_removed=True,
        ndisplay=3,
    ):
        """Show raw data and current segmented_data as colored napari layers."""
        return show_data_and_segmented(
            self.data,
            self.segmented_data,
            raw_colormap=raw_colormap,
            segmented_colormap=segmented_colormap,
            removed_colormap=removed_colormap,
            raw_opacity=raw_opacity,
            segmented_opacity=segmented_opacity,
            removed_opacity=removed_opacity,
            show_removed=show_removed,
            ndisplay=ndisplay,
        )


def remove_holder_tube_angular(
    data,
    start_slice=0,
    stop_slice=None,
    radial_margin=2,
    n_angles=360,
    show_progress=False,
    debug=False,
    background_cutoff=None,
    safe_removal=0,
):
    """PhenoCT angle-dependent holder removal, adapted for local pipeline.

    Source: https://github.com/aus-plant-phenomics-network/phenoct
    (src/phenoct.py, Tube.segment_sample_holder, main at 5181eb81).

    Differences from upstream: conserve full original Z/Y/X shape and
    original slice indices; avoid two additional full-size mask volumes;
    background thresholding and island removal are opt-in because the caller
    already applies its own denoising/island cleanup. No automatic fallback
    to legacy geometry if boundary estimation fails.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected ZYX 3D volume, got {data.shape}")
    if stop_slice is None:
        stop_slice = data.shape[0]
    if not (0 <= start_slice < stop_slice <= data.shape[0]):
        raise ValueError(f"Invalid slice range: [{start_slice}, {stop_slice}) for {data.shape}")
    if n_angles < 16:
        raise ValueError("n_angles must be at least 16")
    if radial_margin < 0:
        raise ValueError("radial_margin must be >= 0")
    if safe_removal < 0:
        raise ValueError("safe_removal must be >= 0")

    # The upstream code uses np.max(volume, axis=0). The slice-wise form
    # below keeps memory bounded when the REK volume is a large memmap.
    projection = np.zeros(data.shape[1:], dtype=data.dtype)
    for z in range(start_slice, stop_slice):
        np.maximum(projection, data[z], out=projection)
    proj = projection.astype(np.float32)
    h, w = proj.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    total = float(proj.sum(dtype=np.float64))
    if total <= 0:
        raise ValueError("Empty projection; holder boundary cannot be detected")
    cy = float(np.sum(yy * proj, dtype=np.float64) / total)
    cx = float(np.sum(xx * proj, dtype=np.float64) / total)

    rr = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx)
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    angle_width = np.pi / n_angles
    inner_radii = np.full(n_angles, np.nan, dtype=np.float32)

    # Preserve the method in upstream PhenoCT: sort the max-projected
    # intensities radially within each angular sector and find the large
    # positive radial gradient followed by a low-gradient inner boundary.
    for i, angle in enumerate(angles):
        sector = np.abs(np.angle(np.exp(1j * (theta - angle)))) < angle_width
        if not np.any(sector):
            continue
        r_vals = rr[sector]
        i_vals = proj[sector]
        order = np.argsort(r_vals)
        r_sorted = r_vals[order]
        prof = gaussian_filter1d(i_vals[order], sigma=2)
        if len(prof) < 5:
            continue
        grad = np.gradient(prof)
        r_start = int(0.25 * len(prof))
        if r_start >= len(prof):
            continue
        tube_idx = r_start + int(np.argmax(grad[r_start:]))
        if tube_idx < 2:
            continue
        baseline = np.median(np.abs(grad[:tube_idx]))
        inner_candidates = np.where(np.abs(grad[:tube_idx]) < baseline)[0]
        if len(inner_candidates):
            inner_radii[i] = r_sorted[inner_candidates[-1]]
        else:
            # In flat-background scans (e.g. near-binary tube intensity),
            # the upstream median-gradient criterion can return no pixels.
            # Use the location of its detected rising edge instead.
            inner_radii[i] = r_sorted[tube_idx]

    valid = np.isfinite(inner_radii)
    if int(valid.sum()) < max(8, n_angles // 10):
        raise RuntimeError(
            f"Only {int(valid.sum())}/{n_angles} valid angular holder boundaries. "
            "Review the input Z range or try method='legacy'."
        )
    med_r = float(np.nanmedian(inner_radii))
    inner_radii[~valid] = med_r
    inner_radii = gaussian_filter1d(inner_radii, sigma=5, mode="wrap")
    if not np.isfinite(inner_radii).all() or float(np.min(inner_radii)) <= radial_margin:
        raise RuntimeError("Invalid holder radii; review debug output")

    # Same angular sector definition as the upstream implementation, but
    # map every pixel to its nearest angle bin to avoid a 360 x H x W mask.
    angle_id = np.floor((theta + np.pi) * n_angles / (2 * np.pi) + 0.5).astype(np.int32)
    angle_id %= n_angles
    remove_mask_2d = rr >= (inner_radii[angle_id] - radial_margin)
    keep_mask_2d = ~remove_mask_2d

    if debug:
        print(
            f"[angular holder removal] centre(x,y)=({cx:.1f},{cy:.1f}), "
            f"inner radius median={med_r:.1f} px, "
            f"range=({inner_radii.min():.1f},{inner_radii.max():.1f}), "
            f"angular sectors with valid estimate={int(valid.sum())}/{n_angles}"
        )

    # Preserve exact input volume shape and Z indexing for cut detection.
    result = np.zeros_like(data)
    zs = range(start_slice, stop_slice)
    if show_progress and tqdm is not None:
        zs = tqdm(zs, desc="Removing holder (angular)")
    for z in zs:
        result[z] = np.where(keep_mask_2d, data[z], 0)

    # Optional PhenoCT background cutoff; leave it disabled in the existing
    # pipeline unless deliberately chosen (another threshold comes next).
    if background_cutoff is not None:
        if background_cutoff <= 0:
            raise ValueError("background_cutoff must be positive or None")
        vals = result[start_slice:stop_slice]
        positive = vals[vals > 0]
        if positive.size == 0:
            raise RuntimeError("No positive voxels remained after holder removal")
        hist, bins = np.histogram(positive, bins=256)
        smooth = gaussian_filter1d(hist.astype(float), sigma=2)
        bg_peak = float(bins[int(np.argmax(smooth))])
        cutoff = bg_peak * background_cutoff
        if debug:
            print(f"[angular holder removal] background peak={bg_peak:.1f}, cutoff={cutoff:.1f}")
        for z in range(start_slice, stop_slice):
            result[z][result[z] < cutoff] = 0

    if safe_removal > 0:
        # Match the pipeline's established 26-connected island cleanup.
        cleaned, _ = threshold_denoise_and_remove_islands(
            result, noise_threshold=1, min_component_size=safe_removal
        )
        result = cleaned

    return result


def remove_holder_tube(
    data,
    start_slice=0,
    stop_slice=None,
    tube_r=160,
    tube_thickness=30,
    sample_step=10,
    outlier_px=12,
    show_progress=False,
):
    """Remove only the holder tube from a REK volume using interpolated centers."""
    if cv2 is None:
        raise ImportError(
            "Tube removal requires OpenCV. Install it with: pip install opencv-python"
        )

    if stop_slice is None:
        stop_slice = data.shape[0]

    if start_slice < 0 or stop_slice > data.shape[0] or stop_slice <= start_slice:
        raise ValueError(
            f"Invalid slice range: start_slice={start_slice}, stop_slice={stop_slice}"
        )

    if sample_step <= 0:
        raise ValueError(f"sample_step must be positive, got {sample_step}")
    if outlier_px <= 0:
        raise ValueError(f"outlier_px must be positive, got {outlier_px}")

    center_x, center_y = estimate_holder_center_trace(
        data,
        start_slice=start_slice,
        stop_slice=stop_slice,
        sample_step=sample_step,
        outlier_px=outlier_px,
    )

    output = np.zeros_like(data)
    iterator = range(start_slice, stop_slice)
    if show_progress and tqdm is not None:
        iterator = tqdm(
            iterator,
            total=stop_slice - start_slice,
            desc="Removing holder tube",
        )

    for z in iterator:
        img = data[z]
        center = (
            int(round(center_x[z - start_slice])),
            int(round(center_y[z - start_slice])),
        )
        mask = make_inner_tube_mask(
            data.shape[1:],
            center=center,
            inner_radius=tube_r - tube_thickness,
        )
        masked = img.copy()
        masked[mask == 0] = 0
        output[z] = masked

    return output


def estimate_holder_center_trace(
    data,
    start_slice,
    stop_slice,
    sample_step=10,
    outlier_px=12,
):
    """Estimate per-slice holder centers from sampled RANSAC fits."""
    sample_indices = np.arange(start_slice, stop_slice, sample_step, dtype=int)
    if sample_indices[-1] != stop_slice - 1:
        sample_indices = np.append(sample_indices, stop_slice - 1)

    fitted_z = []
    fitted_centers = []
    for z in sample_indices:
        try:
            fitted_centers.append(detect_holder_circle(data[z]))
            fitted_z.append(z)
        except RuntimeError:
            continue

    if len(fitted_centers) < 2:
        raise RuntimeError("Not enough holder circle fits to interpolate centers.")

    fitted_z = np.array(fitted_z, dtype=float)
    fitted_centers = np.array(fitted_centers, dtype=float)

    median_center = np.median(fitted_centers, axis=0)
    distances = np.linalg.norm(fitted_centers - median_center, axis=1)
    keep = distances <= outlier_px

    if np.count_nonzero(keep) < 2:
        keep = distances <= np.percentile(distances, 75)
    if np.count_nonzero(keep) < 2:
        raise RuntimeError("Not enough non-outlier holder circle fits.")

    kept_z = fitted_z[keep]
    kept_centers = fitted_centers[keep]
    target_z = np.arange(start_slice, stop_slice, dtype=float)

    center_x = np.interp(target_z, kept_z, kept_centers[:, 0])
    center_y = np.interp(target_z, kept_z, kept_centers[:, 1])

    return center_x, center_y


def filter_by_otsu(
    data,
    otsu_factor=0.8,
    verbose=False,
):
    """
    Apply one global relaxed Otsu threshold while preserving intensities.
    """
    if otsu_factor <= 0:
        raise ValueError(f"otsu_factor must be positive, got {otsu_factor}")

    values = data[data > 0]
    if values.size == 0:
        raise ValueError("Cannot apply Otsu thresholding: no non-zero voxels found.")

    otsu_value = threshold_otsu(values)
    threshold_value = otsu_value * otsu_factor
    output = np.where(data > threshold_value, data, 0)

    if verbose:
        before_nonzero = int(np.count_nonzero(data))
        after_nonzero = int(np.count_nonzero(output))
        removed = before_nonzero - after_nonzero
        removed_pct = (removed / before_nonzero * 100) if before_nonzero else 0.0
        print(
            "Global Otsu filter: "
            f"otsu={otsu_value:.1f}, factor={otsu_factor}, "
            f"threshold={threshold_value:.1f}; "
            f"nonzero {before_nonzero} -> {after_nonzero} "
            f"({removed_pct:.2f}% removed)"
        )

    return output.astype(data.dtype)


def crop_nonzero_data(data):
    """Crop a 2D or 3D array to the bounding box of non-zero values."""
    data = np.asarray(data)
    nonzero_indices = np.nonzero(data)
    if len(nonzero_indices[0]) == 0:
        return data

    slices = tuple(
        slice(axis_indices.min(), axis_indices.max() + 1)
        for axis_indices in nonzero_indices
    )
    return data[slices]


def generate_pseudo_label(
    data,
    threshold=1500,
    min_component_size=1000,
):
    """
    Generate a binary pseudo label by thresholding and removing small islands.

    The connected-component cleanup is 3D, using 26-connectivity.
    """
    if min_component_size <= 0:
        raise ValueError(
            f"min_component_size must be positive, got {min_component_size}"
        )

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {data.shape}")

    threshold_mask = data >= threshold
    _, before_components = ndi.label(threshold_mask)

    cleaned_mask = remove_small_objects(
        threshold_mask.astype(bool),
        min_size=min_component_size,
        connectivity=3,
    )
    _, after_components = ndi.label(cleaned_mask)

    stats = {
        "threshold": float(threshold),
        "min_component_size": int(min_component_size),
        "before_components": int(before_components),
        "after_components": int(after_components),
        "before_voxels": int(np.count_nonzero(threshold_mask)),
        "after_voxels": int(np.count_nonzero(cleaned_mask)),
        "removed_voxels": int(
            np.count_nonzero(threshold_mask) - np.count_nonzero(cleaned_mask)
        ),
        "before_active_slices": int(
            np.count_nonzero(np.any(threshold_mask, axis=(1, 2)))
        ),
        "after_active_slices": int(
            np.count_nonzero(np.any(cleaned_mask, axis=(1, 2)))
        ),
    }

    return cleaned_mask.astype(bool), stats


def threshold_denoise_and_remove_islands(
    data,
    noise_threshold=1500,
    min_component_size=3000,
):
    """
    Threshold a 3D volume and remove small connected components.

    The output preserves original intensity values for voxels in components
    whose size is at least min_component_size.
    """
    if min_component_size <= 0:
        raise ValueError(
            f"min_component_size must be positive, got {min_component_size}"
        )

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {data.shape}")

    threshold_mask = data >= noise_threshold
    _, before_components = ndi.label(threshold_mask)

    cleaned_mask = remove_small_objects(
        threshold_mask.astype(bool),
        min_size=min_component_size,
        connectivity=3,
    )
    _, after_components = ndi.label(cleaned_mask)

    cleaned_data = np.where(cleaned_mask, data, 0).astype(data.dtype)
    stats = {
        "noise_threshold": float(noise_threshold),
        "min_component_size": int(min_component_size),
        "before_components": int(before_components),
        "after_components": int(after_components),
        "before_voxels": int(np.count_nonzero(threshold_mask)),
        "after_voxels": int(np.count_nonzero(cleaned_mask)),
        "removed_voxels": int(
            np.count_nonzero(threshold_mask) - np.count_nonzero(cleaned_mask)
        ),
        "before_active_slices": int(
            np.count_nonzero(np.any(threshold_mask, axis=(1, 2)))
        ),
        "after_active_slices": int(
            np.count_nonzero(np.any(cleaned_mask, axis=(1, 2)))
        ),
    }

    return cleaned_data, stats


def detect_peduncle_boundaries(
    vol,
    smooth_win=21,
    slice_mm=0.084,
    max_len_mm=50.0,
    head_margin_slices=12,
    tail_margin_slices=12,
    rise_threshold=200,
    verbose=False,
):
    """Detect peduncle z-boundaries from the per-slice foreground area curve."""
    mask = vol > 0
    slice_area = mask.sum(axis=(1, 2)).astype(float)
    z_count = slice_area.size

    nz = np.flatnonzero(slice_area > 0)
    if nz.size == 0:
        raise RuntimeError("No plant detected.")

    z_top, z_bot = int(nz[0]), int(nz[-1])
    span = z_bot - z_top + 1

    if smooth_win <= 1:
        area_s = slice_area.copy()
    else:
        kernel = np.ones(smooth_win, dtype=float) / smooth_win
        area_s = np.convolve(slice_area, kernel, mode="same")
    grad = np.diff(area_s)

    search_end = z_top + int(0.6 * span)
    search_end = max(search_end, z_top + 1)
    idx_drop = int(np.argmin(grad[z_top:search_end]) + z_top)

    low_start = idx_drop + 20
    low_end = z_bot - 100
    window_size = 200
    min_variance = float("inf")
    stable_center = None

    for z in range(low_start, max(low_start + 1, low_end - window_size)):
        win = area_s[z : z + window_size]
        var = np.var(win)
        if var < min_variance:
            min_variance = var
            stable_center = z + window_size // 2

    if stable_center is None:
        stable_center = (max(z_top, low_start) + max(z_top, low_end)) // 2
        stable_center = int(np.clip(stable_center, z_top, z_bot))

    rise_search_start = stable_center + 100
    rise_region = grad[rise_search_start : z_bot - 10]

    if rise_region.size > 0:
        idx_rise = int(np.argmax(rise_region) + rise_search_start)
        rise_magnitude = grad[idx_rise]

        if rise_magnitude < rise_threshold:
            s0 = max(0, stable_center - 100)
            s1 = min(z_count, stable_center + 100)
            stable_mean = np.mean(area_s[s0:s1])
            threshold = stable_mean * 1.5

            candidates = np.where(area_s[rise_search_start:z_bot] > threshold)[0]
            if len(candidates) > 0:
                idx_rise = int(candidates[0] + rise_search_start)
            else:
                idx_rise = z_bot - 200
    else:
        idx_rise = z_bot - 200

    spike_base = idx_drop
    flag_node = idx_rise

    z1 = max(z_top, spike_base + head_margin_slices)
    z2 = min(z_bot, flag_node - tail_margin_slices)

    if z2 <= z1:
        z1 = max(z_top, stable_center - 200)
        z2 = min(z_bot, stable_center + 200)

    max_slices = int(np.ceil(max_len_mm / slice_mm))
    if z2 - z1 > max_slices:
        center = (z1 + z2) // 2
        z1 = center - max_slices // 2
        z2 = center + max_slices // 2

    z1 = int(max(z_top, z1))
    z2 = int(min(z_bot + 1, z2))

    if verbose:
        print(
            "Peduncle bounds: "
            f"plant=[{z_top}, {z_bot}], spike_base={spike_base}, "
            f"stable_center={stable_center}, flag_node={flag_node}, "
            f"peduncle=[{z1}, {z2}), len={z2 - z1}"
        )

    return z1, z2


def plot_peduncle_area_profile(vol, z1, z2):
    """Plot foreground area by z with selected peduncle bounds."""
    import matplotlib.pyplot as plt

    slice_area = (vol > 0).sum(axis=(1, 2))
    z_axis = np.arange(slice_area.size)
    plt.figure(figsize=(10, 4))
    plt.plot(z_axis, slice_area, linewidth=1)
    plt.axvline(z1, color="green", linestyle="--", label=f"peduncle start {z1}")
    plt.axvline(z2, color="orange", linestyle="--", label=f"peduncle end {z2}")
    plt.xlim(0, max(0, slice_area.size - 1))
    plt.xlabel("Z slice")
    plt.ylabel("Foreground area")
    plt.title("Peduncle z-range")
    plt.legend()
    plt.tight_layout()
    plt.show()


def extract_main_3d_structure(mask, min_component_size=3000):
    """Keep the largest 3D component and return extraction stats."""
    if min_component_size <= 0:
        raise ValueError(
            f"min_component_size must be positive, got {min_component_size}"
        )

    mask = mask.astype(bool)
    _, before_components = ndi.label(mask)
    cleaned = remove_small_objects(
        mask,
        min_size=min_component_size,
        connectivity=3,
    )

    labels, n_labels = ndi.label(cleaned)
    if n_labels == 0:
        structure_mask = cleaned
    else:
        counts = np.bincount(labels.ravel())
        keep_label = int(np.argmax(counts[1:]) + 1)
        structure_mask = labels == keep_label

    stats = {
        "before_components": int(before_components),
        "after_small_components": int(n_labels),
        "before_voxels": int(np.count_nonzero(mask)),
        "after_main_voxels": int(np.count_nonzero(structure_mask)),
        "before_active_slices": int(np.count_nonzero(np.any(mask, axis=(1, 2)))),
        "after_active_slices": int(
            np.count_nonzero(np.any(structure_mask, axis=(1, 2)))
        ),
    }

    return structure_mask, stats


def calculate_wall_thickness_from_mask(
    peduncle_mask,
    px_xy_mm=0.084,
    closing_radius=1,
    verbose=False,
):
    """Calculate wall thickness per slice from a binary peduncle mask."""
    from skimage.morphology import closing as sk_closing
    from skimage.morphology import disk as sk_disk

    kept_z = []
    thickness_mm = []
    outer_radius_mm = []
    lumen_radius_mm = []
    lumen_area_px = []
    outer_area_px = []
    lumen_mask = np.zeros_like(peduncle_mask, dtype=bool)

    footprint = sk_disk(closing_radius) if closing_radius > 0 else None

    for z in range(peduncle_mask.shape[0]):
        sl0 = peduncle_mask[z].astype(bool)
        if not sl0.any():
            continue

        sl = sk_closing(sl0, footprint) if footprint is not None else sl0
        outer = ndi.binary_fill_holes(sl)
        lumen = outer & (~sl)

        lumen_area = int(lumen.sum())
        if lumen_area <= 0:
            continue

        lumen_mask[z] = lumen
        outer_area = int(outer.sum())
        outer_radius_px = np.sqrt(outer_area / np.pi)
        lumen_radius_px = np.sqrt(lumen_area / np.pi)
        thickness = max((outer_radius_px - lumen_radius_px) * px_xy_mm, 0.0)

        kept_z.append(z)
        thickness_mm.append(thickness)
        outer_radius_mm.append(outer_radius_px * px_xy_mm)
        lumen_radius_mm.append(lumen_radius_px * px_xy_mm)
        lumen_area_px.append(lumen_area)
        outer_area_px.append(outer_area)

    result = {
        "kept_z": np.array(kept_z, dtype=int),
        "thickness_mm": np.array(thickness_mm, dtype=float),
        "outer_radius_mm": np.array(outer_radius_mm, dtype=float),
        "lumen_radius_mm": np.array(lumen_radius_mm, dtype=float),
        "lumen_area_px": np.array(lumen_area_px, dtype=int),
        "outer_area_px": np.array(outer_area_px, dtype=int),
        "lumen_mask": lumen_mask,
    }

    if result["thickness_mm"].size:
        result["mean_thickness_mm"] = float(result["thickness_mm"].mean())
        result["median_thickness_mm"] = float(np.median(result["thickness_mm"]))
    else:
        result["mean_thickness_mm"] = 0.0
        result["median_thickness_mm"] = 0.0

    if verbose:
        print(
            "Wall thickness: "
            f"kept_slices={result['kept_z'].size}, "
            f"mean={result['mean_thickness_mm']:.3f} mm, "
            f"median={result['median_thickness_mm']:.3f} mm"
        )

    return result


def calculate_cross_section_metrics_from_mask(
    peduncle_mask,
    intensity_data=None,
    px_xy_mm=0.084,
    closing_radius=1,
    lumen_quantile=None,
    min_lumen_area_px=50,
    verbose=False,
):
    """Calculate summary cross-section metrics for a 3D binary peduncle mask."""
    from skimage.morphology import closing as sk_closing
    from skimage.morphology import disk as sk_disk

    peduncle_mask = np.asarray(peduncle_mask, dtype=bool)
    if peduncle_mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {peduncle_mask.shape}")

    if intensity_data is None:
        intensity_data = peduncle_mask.astype(float)
    else:
        intensity_data = np.asarray(intensity_data)

    if intensity_data.shape != peduncle_mask.shape:
        raise ValueError(
            f"Intensity data shape {intensity_data.shape} does not match mask shape {peduncle_mask.shape}"
        )

    footprint = sk_disk(closing_radius) if closing_radius > 0 else None
    pixel_area = float(px_xy_mm * px_xy_mm)

    d_out_values = []
    d_in_values = []
    wt_values = []
    csa_values = []
    pr_values = []
    i_values = []
    z_values = []
    wall_mean_intensity_values = []

    for z in range(peduncle_mask.shape[0]):
        slice_mask = peduncle_mask[z]
        if not np.any(slice_mask):
            continue

        closed_slice = (
            sk_closing(slice_mask.astype(bool), footprint)
            if footprint is not None
            else slice_mask.astype(bool)
        )
        outer_mask = ndi.binary_fill_holes(closed_slice).astype(bool)
        outer_area_px = int(np.count_nonzero(outer_mask))
        if outer_area_px <= 0:
            continue

        intensity_slice = intensity_data[z]
        if lumen_quantile is None:
            lumen_mask = outer_mask & (~closed_slice)
            lumen_mask = lumen_mask.astype(bool)
        else:
            outer_values = intensity_slice[outer_mask]
            if outer_values.size == 0:
                continue

            low_threshold = float(np.percentile(outer_values, 100 * lumen_quantile))
            lumen_candidates = (intensity_slice <= low_threshold) & outer_mask
            lumen_candidates = remove_small_objects(
                lumen_candidates.astype(bool),
                min_size=max(1, int(min_lumen_area_px)),
                connectivity=2,
            )

            labels, n_labels = ndi.label(lumen_candidates)
            if n_labels > 0:
                counts = np.bincount(labels.ravel())
                keep_label = int(np.argmax(counts[1:]) + 1)
                lumen_mask = labels == keep_label
            else:
                lumen_mask = np.zeros_like(outer_mask, dtype=bool)

        lumen_area_px = int(np.count_nonzero(lumen_mask))
        if lumen_area_px <= 0:
            continue

        wall_mask = outer_mask & ~lumen_mask
        wall_area_px = int(np.count_nonzero(wall_mask))
        if wall_area_px <= 0:
            continue

        d_out_mm = 2.0 * np.sqrt(outer_area_px / np.pi) * px_xy_mm
        d_in_mm = 2.0 * np.sqrt(lumen_area_px / np.pi) * px_xy_mm
        wt_mm = max((d_out_mm - d_in_mm) / 2.0, 0.0)
        csa_mm2 = max((outer_area_px - lumen_area_px) * pixel_area, 0.0)
        pr = lumen_area_px / float(outer_area_px) if outer_area_px > 0 else 0.0

        yy, xx = np.indices(outer_mask.shape, dtype=float)
        x_coords = xx * px_xy_mm
        y_coords = yy * px_xy_mm
        x_centroid = float(np.mean(x_coords[wall_mask]))
        y_centroid = float(np.mean(y_coords[wall_mask]))

        i_x = float(np.sum((y_coords[wall_mask] - y_centroid) ** 2) * pixel_area)
        i_y = float(np.sum((x_coords[wall_mask] - x_centroid) ** 2) * pixel_area)
        i_mm4 = i_x + i_y
        z_mm3 = float(i_mm4 / max(d_out_mm / 2.0, 1e-9))
        wall_mean_intensity = float(np.mean(intensity_slice[wall_mask]))

        d_out_values.append(d_out_mm)
        d_in_values.append(d_in_mm)
        wt_values.append(wt_mm)
        csa_values.append(csa_mm2)
        pr_values.append(pr)
        i_values.append(i_mm4)
        z_values.append(z_mm3)
        wall_mean_intensity_values.append(wall_mean_intensity)

    if not d_out_values:
        return {
            "valid_slices": 0,
            "d_out_mm": float("nan"),
            "d_in_mm": float("nan"),
            "wt_mm": float("nan"),
            "csa_mm2": float("nan"),
            "pr": float("nan"),
            "i_mm4": float("nan"),
            "z_mm3": float("nan"),
            "wall_mean_intensity": float("nan"),
        }

    result = {
        "valid_slices": int(len(d_out_values)),
        "d_out_mm": float(np.nanmean(d_out_values)),
        "d_in_mm": float(np.nanmean(d_in_values)),
        "wt_mm": float(np.nanmean(wt_values)),
        "csa_mm2": float(np.nanmean(csa_values)),
        "pr": float(np.nanmean(pr_values)),
        "i_mm4": float(np.nanmean(i_values)),
        "z_mm3": float(np.nanmean(z_values)),
        "wall_mean_intensity": float(np.nanmean(wall_mean_intensity_values)),
    }

    return result


def plot_wall_thickness_profile(result):
    """Plot wall thickness by z slice."""
    import matplotlib.pyplot as plt

    kept_z = result["kept_z"]
    thickness = result["thickness_mm"]
    if kept_z.size == 0:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(kept_z, thickness, linewidth=1.2)
    plt.xlabel("Z slice")
    plt.ylabel("Wall thickness (mm)")
    plt.title("Peduncle wall thickness per slice")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_pixel_intensity_profile(result):
    """Plot per-slice pixel intensity inside the extracted structure."""
    import matplotlib.pyplot as plt

    kept_z = result["kept_z"]
    if kept_z.size == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(kept_z, result["slice_mean_intensity"], linewidth=1.2)
    axes[0].set_ylabel("Mean intensity")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(kept_z, result["slice_sum_intensity"], linewidth=1.2)
    axes[1].set_xlabel("Z slice")
    axes[1].set_ylabel("Sum intensity")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Peduncle pixel intensity per slice")
    fig.tight_layout()
    plt.show()


def plot_intensity_histogram(
    data,
    bins=256,
    exclude_zero=True,
    intensity_range=None,
    log_y=False,
    title="Data intensity histogram",
):
    """Plot a grayscale intensity histogram for raw volume data."""
    import matplotlib.pyplot as plt

    values = np.asarray(data).ravel()
    if exclude_zero:
        values = values[values > 0]

    if values.size == 0:
        raise ValueError("Cannot plot intensity histogram: no voxels available.")

    counts, bin_edges = np.histogram(
        values,
        bins=bins,
        range=intensity_range,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color="0.25",
        edgecolor="0.25",
    )
    ax.set_xlabel("Gray value")
    ax.set_ylabel("Voxel count")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.show()

    return counts, bin_edges


def show_wall_thickness_debug(data, structure_data, lumen_data):
    """Show raw data, extracted peduncle structure, and lumen in napari."""
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "Visualization requires napari. Install it with: pip install napari"
        ) from exc

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        data,
        name="data",
        colormap="gray",
        opacity=0.25,
        blending="additive",
    )
    viewer.add_image(
        structure_data,
        name="peduncle_structure",
        colormap="green",
        opacity=0.7,
        blending="additive",
    )
    viewer.add_image(
        lumen_data,
        name="lumen",
        colormap="magenta",
        opacity=0.8,
        blending="translucent_no_depth",
        contrast_limits=(0, 255),
    )
    return viewer


def show_data_and_segmented(
    data,
    segmented_data,
    raw_colormap="cyan",
    segmented_colormap="yellow",
    removed_colormap="red",
    raw_opacity=0.35,
    segmented_opacity=0.65,
    removed_opacity=0.5,
    show_removed=True,
    ndisplay=3,
):
    """Open napari with raw, segmented, and removed data shown separately."""
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "Visualization requires napari. Install it with: pip install napari"
        ) from exc

    if data.shape != segmented_data.shape:
        raise ValueError(
            f"data and segmented_data must have the same shape, got "
            f"{data.shape} and {segmented_data.shape}"
        )

    viewer = napari.Viewer(ndisplay=ndisplay)
    viewer.add_image(
        data,
        name="data",
        colormap=raw_colormap,
        opacity=raw_opacity,
        blending="additive",
    )
    viewer.add_image(
        segmented_data,
        name="segmented_data",
        colormap=segmented_colormap,
        opacity=segmented_opacity,
        blending="additive",
    )

    if show_removed:
        removed_data = np.where(segmented_data == 0, data, 0).astype(data.dtype)
        viewer.add_image(
            removed_data,
            name="removed_data",
            colormap=removed_colormap,
            opacity=removed_opacity,
            blending="additive",
        )

    return viewer


def detect_holder_circle(v_slice):
    """Detect the holder tube circle with edge detection and RANSAC fitting."""
    h, w = v_slice.shape
    img8 = normalize_slice_to_uint8(v_slice)
    blurred = cv2.GaussianBlur(img8, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    ys, xs = np.nonzero(edges)
    if xs.size < 20:
        raise RuntimeError("No sample holder circle found.")

    image_center = np.array([w / 2.0, h / 2.0])
    edge_points = np.column_stack([xs, ys]).astype(float)
    radii_from_image_center = np.linalg.norm(edge_points - image_center, axis=1)

    min_radius = 150
    max_radius = min(h, w) / 2.0
    candidate_points = edge_points[
        (radii_from_image_center >= min_radius * 0.6)
        & (radii_from_image_center <= max_radius * 1.1)
    ]
    if candidate_points.shape[0] < 20:
        candidate_points = edge_points

    max_points = 8000
    if candidate_points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(candidate_points.shape[0], size=max_points, replace=False)
        candidate_points = candidate_points[keep]

    try:
        model, inliers = ransac(
            candidate_points,
            CircleModel,
            min_samples=3,
            residual_threshold=2,
            max_trials=500,
        )
    except ValueError as exc:
        raise RuntimeError("No sample holder circle found.") from exc

    if model is None or model.center is None or model.radius is None:
        raise RuntimeError("No sample holder circle found.")

    x, y = model.center
    r = model.radius
    if not (min_radius <= r <= max_radius * 1.1):
        raise RuntimeError(
            f"Detected holder radius is out of range: radius={r:.1f}"
        )

    return int(round(x)), int(round(y))


def normalize_slice_to_uint8(v_slice):
    """Robustly normalize a CT slice to uint8 for edge detection."""
    positive = v_slice[v_slice > 0]
    if positive.size:
        lo, hi = np.percentile(positive, (1, 99.5))
    else:
        lo, hi = int(v_slice.min()), int(v_slice.max())

    if hi <= lo:
        hi = lo + 1

    normalized = (v_slice.astype(np.float32) - lo) / (hi - lo)
    normalized = np.clip(normalized, 0, 1)
    return (normalized * 255).astype(np.uint8)


def make_inner_tube_mask(shape, center, inner_radius):
    """Create a filled uint8 mask for the safe inner tube region."""
    if inner_radius <= 0:
        raise ValueError(f"inner_radius must be positive, got {inner_radius}")

    h, w = shape
    x, y = center
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), int(inner_radius), 255, 1)

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(mask, flood_mask, (x, y), 255)

    return mask
