'''
Clip datasets

Note: Dataset CRS is CRS - EPSG:**** 
All GIS file must be in this projection.
'''
#%% IMPORT
from pathlib import Path
import geopandas
import os
from shapely.geometry import box
import datetime
from shapely.geometry import Polygon
#%% FILL BEFORE RUNNING
data_types = ['Raster', 'Vector'] 
save_boolean = True
buffer_widths = [30] #[15, 25, 30, 35, 45]

main_path=r'p:\11208012-011-nabaripoma\Data'

datasets = ['Grondwaterstand.gdb', 'Bodemkaart/Bodemkaart.gdb', 'BBG/BBG.gdb', 'BAG/BAG.gdb', 'BAG/BAG.gdb',
            'BRT/TOP10NL.gdb', 'BRT/TOP10NL.gdb', 'CBS_Vierkantstatistieken/CBS_Vierkantstatistieken_2018_v1.gdb',
            'Woningtypering.gdb', 'OSR_etc', 'polderpeilen']
layers = ['Gemiddelde_Laagste_Grondwaterstand_Huidig', 'Bodemkaart', 'BBG2012', 'Pand', 'Verblijfsobject',
            'WEGDEEL_VLAK', 'WEGDEEL_HARTLIJN', 'CBS_VK100_2018_v1', 'Woningtypering', 'Bruto_Heel_NL.shp', 'Peilgebieden_all_zp_wp2.shp']

#%% Function
def clip_gis_database(data_types, datasets, layers, buffer_width, save_boolean=True):
    # project_filenames = list(Path(
        # r'p:\11205151-wu2c-kostendata/ingevulde_excels_trace/' + type_company + '/' + company_name + '/GIS_trace').glob(
        # '*.shp'))
    filenames_location = Path(os.path.abspath(os.path.join(main_path, data_type)))
    if not os.path.exists(filenames_location):
        print(filenames_location, 'does not exist')
        
    project_filenames = list(filenames_location.glob('*.shp'))
    dataset_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..','11205151-wu2c-kostendata',
                                                    'main_datasets')))
    output_dir = os.path.abspath(os.path.join(main_path, 'Scripts','output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('New output directory created: ' + output_dir)
        # project_filenames = list(Path(r'p:\11205151-wu2c-kostendata/ingevulde_excels_trace/'+ type_company + '/' + company_name + '/GIS_trace').glob('*.shp'))
        # dataset_dir = Path(r'p:\11205151-wu2c-kostendata\main_datasets')
        # output_dir = Path(r'output/' + company_name)
     
    for project_name in project_filenames:
        print('Data: ' + project_name.stem)
        
        dir_name = project_name.stem + '_' + str(buffer_width)
        project_output_dir = os.path.abspath(os.path.join(output_dir, dir_name))
        
        if not os.path.exists(project_output_dir):
            os.mkdir(project_output_dir)
            print('New project output directory created: ' + project_output_dir)

            #Calculation is only completed if project doesn exist yet
            trace = geopandas.read_file(project_name)
            buffered_trace = trace.buffer(buffer_width)
            polygon = Polygon([(0, 0), (0, 90), (180, 90), (180, 0), (0, 0)]) #22.7,89.2 - 22.7, 89.6, - 22.4, 89.2,  
            poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)
            
            fp = os.path.join(project_output_dir, project_name.stem+'_buffered.shp' )
            buffered_trace.to_file(fp)
            # buffered_trace.to_file(project_output_dir / (project_name.stem+'_buffered.shp'))
            
            for dataset, layer in zip(datasets, layers):
                
                print('\t {}'.format(dataset))
                print('\t\t {}'.format(layer))
                dataset_path = os.path.join(dataset_dir, dataset)
                # dataset_path = Path(dataset_dir) / dataset
                print('\t\t\treading')
    
    
                if Path(dataset_path).suffix == ".gdb":
                    # gdf = geopandas.read_file(dataset_path, layer=layer, bbox=tuple(buffered_trace.total_bounds))
                    gdf = geopandas.read_file(dataset_path, layer=layer, bbox=box(*buffered_trace.total_bounds))
                else:
                    # shapefile_path = dataset_path / layer
                    shapefile_path = os.path.join(dataset_path, layer)
                    # gdf = geopandas.read_file(shapefile_path, bbox=tuple(buffered_trace.total_bounds))
                    gdf = geopandas.read_file(shapefile_path, bbox=box(*buffered_trace.total_bounds))
    
                if gdf.geom_type.all()=='Polygon' or gdf.geom_type.all()=='MultiPolygon':
                    gdf['geometry'] = gdf.geometry.buffer(0)
    
                print('\t\t\tclipping')
                clipped = geopandas.clip(gdf, buffered_trace)
    
                filename = project_name.stem + '_' + layer + '.shp'
                if save_boolean:
                    if clipped.empty:
                        print('\t\t\t', project_name.stem, layer, 'is empty')
                    else:
                        fp = os.path.join(project_output_dir, filename)
                        # clipped.to_file(project_output_dir / filename)
                        clipped.to_file(fp)
        else:
            print('Output directory already present - skipping this project')
            print(' ')

#%% Loop
for buffer_width in buffer_widths:
    print('buffer width: ' + str(buffer_width))
    for i in range(len(names_company)):
        print('Company: ' + names_company[i])
        print (datetime.datetime.now())
        clip_gis_database(names_company[i], types_company[i], datasets, layers, buffer_width, save_boolean)