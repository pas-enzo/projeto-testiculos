/**
 * Export every annotation stored in QuPath .qpdata files to GeoJSON, without opening the images.
 *
 * Usage (from the repository root):
 *   & "D:\QuPath\QuPath-0.7.0 (console).exe" script scripts/qupath/export_geojson.groovy `
 *       -a wsi_annotations -a outputs/annotations
 *
 * For each <slide>.qpdata it writes:
 *   <slide>.geojson    FeatureCollection with all objects (pixel coordinates at full resolution)
 *   <slide>.meta.json  image metadata stored in the .qpdata header (size, pyramid levels, pixel size)
 */
import com.google.gson.reflect.TypeToken
import qupath.lib.io.GsonTools
import qupath.lib.io.PathIO
import qupath.lib.io.PathIO.GeoJsonExportOptions

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

Path inputDir = Paths.get(args.size() > 0 ? args[0] : 'wsi_annotations')
Path outputDir = Paths.get(args.size() > 1 ? args[1] : 'outputs/annotations')
Files.createDirectories(outputDir)

def qpdataFiles = Files.list(inputDir).withCloseable { stream ->
    stream.filter { it.fileName.toString().endsWith('.qpdata') }.sorted().toList()
}
if (qpdataFiles.isEmpty())
    throw new IllegalArgumentException("No .qpdata files found in ${inputDir.toAbsolutePath()}")

for (Path qpdata : qpdataFiles) {
    Map header
    new ObjectInputStream(new BufferedInputStream(Files.newInputStream(qpdata))).withCloseable { ois ->
        ois.readUTF()
        header = GsonTools.getInstance().fromJson(ois.readObject() as String, new TypeToken<Map<String, Object>>() {}.getType())
    }
    Map metadata = header.server.metadata as Map
    String imageName = metadata.name
    String slideId = imageName.replaceFirst(/\.[^.]+$/, '')

    def hierarchy = PathIO.readHierarchy(qpdata)
    def objects = hierarchy.getAllObjects(false)

    PathIO.exportObjectsAsGeoJSON(
            outputDir.resolve(slideId + '.geojson'), objects,
            GeoJsonExportOptions.FEATURE_COLLECTION, GeoJsonExportOptions.PRETTY_JSON)

    def meta = [
            slide_id     : slideId,
            image_name   : imageName,
            qpdata_file  : qpdata.fileName.toString(),
            qupath_version: header.qupathVersion,
            width        : metadata.width,
            height       : metadata.height,
            levels       : metadata.levels,
            pixel_width_um : metadata.pixelCalibration?.pixelWidth?.value,
            pixel_height_um: metadata.pixelCalibration?.pixelHeight?.value,
            magnification: metadata.magnification,
            n_objects    : objects.size(),
    ]
    Files.writeString(outputDir.resolve(slideId + '.meta.json'), GsonTools.getInstance(true).toJson(meta))

    def counts = objects.countBy { "${it.pathClass ?: 'Unclassified'} / ${it.ROI?.roiName ?: 'no ROI'}" }
    println "${qpdata.fileName} -> ${slideId}.geojson (${objects.size()} objects)"
    counts.sort().each { k, v -> println "    ${k}: ${v}" }
}
