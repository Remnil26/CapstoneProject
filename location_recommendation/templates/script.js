import 'ol/ol.css';
import {apply} from 'ol-mapbox-style';

const map = apply('map-container', './data/bright.json');
      
      
      var rome = new ol.Feature({
        geometry: new ol.geom.Point(ol.proj.fromLonLat([73.8213502, 18.5020578]))
      });

      var london = new ol.Feature({
        geometry: new ol.geom.Point(ol.proj.fromLonLat([73.8161631, 18.54761592]))
      });

      var madrid = new ol.Feature({
        geometry: new ol.geom.Point(ol.proj.fromLonLat([73.792175, 18.5574802]))
      });

      rome.setStyle(new ol.style.Style({
        image: new ol.style.Icon( /** @type {olx.style.IconOptions} */ ({
          color: '#8959A8',
          crossOrigin: 'anonymous',
          src: 'https://openlayers.org/en/v4.6.5/examples/data/dot.png'
        }))
      }));

      london.setStyle(new ol.style.Style({
        image: new ol.style.Icon( /** @type {olx.style.IconOptions} */ ({
          color: '#4271AE',
          crossOrigin: 'anonymous',
          src: 'https://openlayers.org/en/v4.6.5/examples/data/dot.png'
        }))
      }));

      madrid.setStyle(new ol.style.Style({
        image: new ol.style.Icon( /** @type {olx.style.IconOptions} */ ({
          color: [113, 140, 0],
          crossOrigin: 'anonymous',
          src: 'https://openlayers.org/en/v4.6.5/examples/data/dot.png'
        }))
      }));


      var vectorSource = new ol.source.Vector({
        features: [rome, london, madrid]
      });



      var lamarin = ol.proj.fromLonLat([73.8213502, 18.5020578]);
      var view = new ol.View({
        center: lamarin,
        zoom: 15
      });

      var vectorSource = new ol.source.Vector({});
      var places = [
        [73.8213502, 18.5020578, 'https://openlayers.org/en/v4.6.5/examples/data/icon.png', [255, 0, 0]],
        [73.8161631, 18.54761592, 'https://openlayers.org/en/v4.6.5/examples/data/icon.png', [255, 0, 0]],
        [73.792175, 18.5574802, 'https://openlayers.org/en/v4.6.5/examples/data/icon.png', [255, 0, 0] ],
      ];

      var features = [];
      for (var i = 0; i < places.length; i++) {
        var iconFeature = new ol.Feature({
          geometry: new ol.geom.Point(ol.proj.transform([places[i][0], places[i][1]], 'EPSG:4326', 'EPSG:3857')),
        });


        /* rome.setStyle(new ol.style.Style({
            image: new ol.style.Icon(({
             color: '#8959A8',
             crossOrigin: 'anonymous',
             src: 'https://openlayers.org/en/v4.6.5/examples/data/dot.png'
            }))
          })); */

        var iconStyle = new ol.style.Style({
          image: new ol.style.Icon({
            anchor: [0.5, 0.5],
            anchorXUnits: 'fraction',
            anchorYUnits: 'fraction',
            src: places[i][2],
            color: places[i][3],
            crossOrigin: 'anonymous',
          })
        });
        iconFeature.setStyle(iconStyle);
        vectorSource.addFeature(iconFeature);
      }



      var vectorLayer = new ol.layer.Vector({
        source: vectorSource,
        updateWhileAnimating: true,
        updateWhileInteracting: true,
        /* style: function(feature, resolution) {
         iconStyle.getImage().setScale(map.getView().getResolutionForZoom(18) / resolution);
          return iconStyle; 
        } */
      });

      var map = new ol.Map({
        target: 'map',
        view: view,
        layers: [
          new ol.layer.Tile({
            preload: 3,
            source: new ol.source.OSM(),
          }),
          vectorLayer,
        ],
        loadTilesWhileAnimating: true,
      });


      /* map.once('postrender', function(event) {
        view.animate({
          center: lamarin,
          zoom: 17,
          duration: 10000,
          mapTypeId: 'roadmap',
        });
      }); */