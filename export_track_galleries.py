"""Export the selected video-track galleries as self-contained offline HTML."""
from pathlib import Path
from html import unescape
from html.parser import HTMLParser
import base64
import mimetypes
import re
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ForensicShare' / 'Video_Cluster_Galleries'
SOURCES = {
    'person_video_clusters.html': 'outputs/clustering/leiden_track_centroid_097/person/gallery_track/leiden_track_gallery.html',
    'object_video_clusters.html': 'outputs/clustering/leiden_object_track_centroid_097/object/gallery_track/leiden_track_gallery.html',
}
VIEWER = '''
<dialog id="offline-image-viewer" style="max-width:96vw;max-height:96vh;background:#111;color:white;border:1px solid #777">
<button id="offline-image-close" type="button" style="display:block;margin-bottom:8px">Close / Esc</button>
<img id="offline-image-full" alt="Enlarged crop" style="max-width:90vw;max-height:85vh;object-fit:contain">
</dialog>
<script>
const viewer = document.getElementById('offline-image-viewer');
document.addEventListener('click', event => {
  const anchor = event.target.closest('a[data-embedded-image]');
  if (!anchor) return;
  event.preventDefault();
  document.getElementById('offline-image-full').src = anchor.querySelector('img').src;
  viewer.showModal();
});
document.getElementById('offline-image-close').onclick = () => viewer.close();
viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
</script>
'''


class Check(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img' and attrs.get('src', '').startswith('data:image/'):
            raw = base64.b64decode(attrs['src'].split(',', 1)[1], validate=True)
            if not raw:
                raise ValueError('Empty image')
            self.images += 1
        for key in ('src', 'href', 'poster'):
            value = attrs.get(key, '')
            if value and not value.startswith(('data:', '#')):
                raise ValueError(f'External dependency: {value}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, relative in SOURCES.items():
        source = ROOT / relative
        text = source.read_text(encoding='utf-8-sig')
        image_count = 0

        def replace(match):
            nonlocal image_count
            key, quote, value = match.groups()
            if not value.startswith('assets/'):
                return match.group(0)
            asset = (source.parent / unescape(value)).resolve()
            asset.relative_to((source.parent / 'assets').resolve())
            if key.lower() == 'href':
                return 'href="#offline-image-viewer" data-embedded-image="true"'
            raw = asset.read_bytes()
            mime = mimetypes.guess_type(asset.name)[0] or 'image/jpeg'
            image_count += 1
            return f'{key}={quote}data:{mime};base64,{base64.b64encode(raw).decode()}{quote}'

        text = re.sub(r'''\b(src|href)=(['"])(.*?)\2''', replace, text, flags=re.I)
        text = text.replace('</body>', VIEWER + '</body>')
        validator = Check()
        validator.feed(text)
        assert validator.images == image_count and image_count > 0
        assert 'file:///' not in text and 'C:\\Users\\' not in text
        target = OUT / name
        target.write_text(text, encoding='utf-8')
        print(name, 'embedded images=', image_count, 'bytes=', target.stat().st_size)

    (OUT / 'README.txt').write_text(
        'Open person_video_clusters.html or object_video_clusters.html in Chrome/Edge.\n'
        'Each HTML contains its own images and works offline. Click an image to enlarge.\n'
        'These are the existing gallery samples, not all DB points or full videos.\n'
        'No Qdrant server, original project directory, or assets folder is required.\n',
        encoding='utf-8',
    )
    archive = OUT / 'Video_Cluster_Galleries.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name in [*SOURCES, 'README.txt']:
            z.write(OUT / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print('ZIP verified:', archive, archive.stat().st_size)


if __name__ == '__main__':
    main()
