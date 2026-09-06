from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os, uuid, json
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
ALLOWED_EXTENSIONS = {'tif','tiff','jpg','jpeg','png'}
app = Flask(__name__)
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, OUTPUT_FOLDER=OUTPUT_FOLDER, MAX_CONTENT_LENGTH=100*1024*1024)
for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, STATIC_FOLDER): os.makedirs(folder, exist_ok=True)

def ext(name): return name.rsplit('.',1)[1].lower() if '.' in name else ''
def allowed_file(name): return ext(name) in ALLOWED_EXTENSIONS

def valid_values(data, nodata):
    mask = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata): mask &= data != nodata
    return mask, data[mask]

def stats(data, nodata):
    mask, vals = valid_values(data, nodata)
    if vals.size == 0: raise ValueError('No valid raster values were found.')
    return mask, {
        'min': float(vals.min()), 'max': float(vals.max()), 'mean': float(vals.mean()),
        'median': float(np.median(vals)), 'std': float(vals.std()),
        'valid_count': int(vals.size), 'nodata_count': int(data.size - vals.size),
        'nodata_percent': float((data.size-vals.size)*100/data.size)
    }

def convert_image_to_tiff(src_path, original_name):
    image = Image.open(src_path)
    image = image.convert('RGBA') if image.mode in ('P','LA') else image
    arr = np.asarray(image)
    if arr.ndim == 2: arr = arr[np.newaxis, ...]
    elif arr.ndim == 3: arr = np.moveaxis(arr, 2, 0)
    else: raise ValueError('Unsupported image structure.')
    # JPEG/PNG are normally non-georeferenced; use pixel-space transform only.
    h, w = arr.shape[1], arr.shape[2]
    dtype = arr.dtype
    if np.issubdtype(dtype, np.integer): out_dtype = dtype
    else: out_dtype = 'float32'; arr = arr.astype('float32')
    stem = os.path.splitext(secure_filename(original_name))[0]
    standardized = f'{stem}_standardized.tif'
    out_path = os.path.join(UPLOAD_FOLDER, standardized)
    profile = {'driver':'GTiff','height':h,'width':w,'count':arr.shape[0],'dtype':np.dtype(out_dtype).name,
               'transform':from_origin(0, h, 1, 1), 'crs':None, 'compress':'deflate'}
    with rasterio.open(out_path,'w',**profile) as dst: dst.write(arr.astype(out_dtype))
    return out_path, standardized, arr.shape[0], w, h

def make_preview(tif_path, stem, normalized=False):
    """
    Create a scientific raster preview.

    Original:
        Shows the original raster/image.

    Normalized:
        Shows the actual normalized pixel values from 0 to 1
        using a scientific colour scale.
    """

    suffix = "normalized" if normalized else "original"
    out = os.path.join(
        STATIC_FOLDER,
        f"{stem}_{suffix}.png"
    )

    with rasterio.open(tif_path) as src:

        data = src.read().astype("float32")

        # -----------------------------------------
        # NORMALIZED RASTER PREVIEW
        # -----------------------------------------
        if normalized:

            # Use Band 1 for scientific visualization
            band = data[0]

            # Normalized data should already be between 0 and 1
            band = np.clip(band, 0, 1)

            fig, ax = plt.subplots(
                figsize=(8, 5.5),
                dpi=160
            )

            image = ax.imshow(
                band,
                cmap="viridis",
                vmin=0,
                vmax=1
            )

            ax.set_title(
                "NORMALIZED RASTER — PIXEL VALUES (0–1)",
                fontsize=13,
                fontweight="bold"
            )

            ax.set_xlabel("Pixel Column")
            ax.set_ylabel("Pixel Row")

            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            colorbar.set_label(
                "Normalized Pixel Value",
                fontsize=10
            )

            ax.text(
                0.02,
                -0.12,
                "Min = 0.0000     |     Max = 1.0000     |     Method = Min-Max",
                transform=ax.transAxes,
                fontsize=9
            )

            fig.tight_layout()

            fig.savefig(
                out,
                bbox_inches="tight"
            )

            plt.close(fig)

        # -----------------------------------------
        # ORIGINAL RASTER PREVIEW
        # -----------------------------------------
        else:

            # RGB image
            if data.shape[0] >= 3:

                rgb = np.moveaxis(
                    data[:3],
                    0,
                    2
                )

                # Display RGB using percentile stretch
                valid = np.isfinite(rgb).all(axis=2)

                if valid.any():

                    low, high = np.nanpercentile(
                        rgb[valid],
                        [2, 98]
                    )

                else:
                    low, high = 0, 255

                rgb = np.clip(
                    (rgb - low) /
                    (high - low + 1e-12),
                    0,
                    1
                )

                plt.imsave(
                    out,
                    rgb
                )

            # Single-band raster
            else:

                band = data[0]

                valid = np.isfinite(band)

                if valid.any():

                    vmin = np.nanmin(
                        band[valid]
                    )

                    vmax = np.nanmax(
                        band[valid]
                    )

                else:

                    vmin = 0
                    vmax = 1

                plt.imsave(
                    out,
                    band,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax
                )

    return f"/static/{os.path.basename(out)}"
def make_histogram(tif_path, stem, normalized=False):
    out=os.path.join(STATIC_FOLDER,f'{stem}_{"normalized" if normalized else "original"}_hist.png')
    with rasterio.open(tif_path) as src:
        data=src.read().astype('float32'); mask, vals=valid_values(data,src.nodata)
        vals=vals[np.isfinite(vals)]
    if vals.size:
        sample=vals if vals.size<=100000 else np.random.default_rng(42).choice(vals,100000,replace=False)
        fig,ax=plt.subplots(figsize=(7,3.2)); ax.hist(sample,bins=40); ax.set_xlabel('Pixel value'); ax.set_ylabel('Frequency'); ax.grid(alpha=.18); fig.tight_layout(); fig.savefig(out,dpi=140); plt.close(fig)
    return f'/static/{os.path.basename(out)}'

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'raster' not in request.files: return jsonify(success=False,error='Please select a raster file.'),400
    file=request.files['raster']
    if not file.filename: return jsonify(success=False,error='No file selected.'),400
    if not allowed_file(file.filename): return jsonify(success=False,error='Supported formats: TIFF, GeoTIFF, JPG, JPEG and PNG.'),400
    original=secure_filename(file.filename); file_id=uuid.uuid4().hex[:12]; original_path=os.path.join(UPLOAD_FOLDER,f'{file_id}_{original}')
    try:
        file.save(original_path); e=ext(original)
        standardized_path=original_path; standardized_name=original; converted=False
        if e in {'jpg','jpeg','png'}:
            standardized_path, standardized_name,_,_,_=convert_image_to_tiff(original_path, original); converted=True
        with rasterio.open(standardized_path) as src:
            data=src.read().astype('float32'); mask,s=stats(data,src.nodata)
            band_stats=[]
            for i in range(src.count):
                _,bs=stats(data[i:i+1],src.nodata); band_stats.append(bs)
            preview=make_preview(standardized_path,file_id,False)
            hist=make_histogram(standardized_path,file_id,False)
            session={'file_id':file_id,'original_name':original,'standardized_name':standardized_name,'path':standardized_path,'converted':converted}
            with open(os.path.join(UPLOAD_FOLDER,f'{file_id}.json'),'w') as f: json.dump(session,f)
            return jsonify(success=True,file_id=file_id,filename=original,format=e.upper(),standardized_format='TIFF',converted=converted,
                width=src.width,height=src.height,bands=src.count,crs=str(src.crs) if src.crs else 'Not defined',georeferenced=bool(src.crs or src.transform != from_origin(0,src.height,1,1)),
                transform=list(src.transform)[:6],pixel_size=[abs(src.transform.a),abs(src.transform.e)],bounds=list(src.bounds),dtype=str(src.dtypes[0]),nodata=src.nodata,
                min_value=s['min'],max_value=s['max'],mean=s['mean'],median=s['median'],std=s['std'],valid_count=s['valid_count'],nodata_count=s['nodata_count'],nodata_percent=s['nodata_percent'],preview_url=preview,histogram_url=hist)
    except Exception as e: return jsonify(success=False,error=f'Raster reading failed: {e}'),500

@app.route('/normalize',methods=['POST'])
def normalize():
    body=request.get_json() or {}; fid=secure_filename(body.get('file_id',''))
    meta=os.path.join(UPLOAD_FOLDER,f'{fid}.json')
    if not os.path.exists(meta): return jsonify(success=False,error='Raster session was not found.'),404
    try:
        with open(meta) as f: session=json.load(f)
        input_path=session['path']; stem=os.path.splitext(secure_filename(session['original_name']))[0]
        output_name=f'{stem}_normalized.tif'; output_path=os.path.join(OUTPUT_FOLDER,output_name)
        with rasterio.open(input_path) as src:
            data=src.read().astype('float32'); mask,s=stats(data,src.nodata)
            if s['max']==s['min']: normalized=np.zeros_like(data,dtype='float32')
            else: normalized=np.clip((data-s['min'])/(s['max']-s['min']),0,1).astype('float32')
            normalized[~mask]=0
            profile=src.profile.copy(); profile.update(dtype='float32',count=src.count,nodata=0,compress='deflate')
            with rasterio.open(output_path,'w',**profile) as dst: dst.write(normalized)
            # Create a Windows-viewable normalized TIFF
        viewable_name = f"{stem}_normalized_viewable.tif"
        viewable_path = os.path.join(OUTPUT_FOLDER, viewable_name)

        viewable = np.clip(normalized * 255, 0, 255).astype("uint8")

        view_profile = profile.copy()
        view_profile.update(dtype="uint8", nodata=0)

        with rasterio.open(viewable_path, "w", **view_profile) as dst:
            dst.write(viewable)
                    
            preview = make_preview(output_path, fid, True)
            hist = make_histogram(output_path, fid, True)
            _, ns = stats(normalized, 0)

            geo = bool(src.crs) or (
                session.get('converted') is False
                and src.transform != from_origin(0, src.height, 1, 1)
            )

            bounds = [
                [src.bounds.bottom, src.bounds.left],
                [src.bounds.top, src.bounds.right]
            ] if src.crs else None

            return jsonify(
                success=True,
                output_name=output_name,
                download_url=f'/download/{viewable_name}',
                original_min=s['min'],
                original_max=s['max'],
                normalized_min=ns['min'],
                normalized_max=ns['max'],
                georeferenced=geo,
                standardized_format='TIFF',
                method='Min-Max Normalization',
                preview_url=f'/static/{fid}_normalized.png',
                histogram_url=f'/static/{fid}_normalized_hist.png'
            )

    except Exception as e:
        return jsonify(
            success=False,
            error=f'Normalization failed: {e}'
        ), 500


@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        secure_filename(filename),
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)
