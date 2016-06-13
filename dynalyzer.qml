import QtQuick 2.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

import org.dynalyzer 1.0


ApplicationWindow {
	id: app
	visible: true
	width: 1000
	height: 800
	title: "Dynalyzer"
	
	TabView {
		id: tabview
		anchors.fill: parent
	
		Tab {
			id: tab
			title: folder
			property string folder: "/home/tuukka/measurement data/dynamat/20160305 Molmat/20160503_0425"
			
			AnalysisView {
				analyzer: analyzer
				hpAnalyzer: hpAnalyzer
				measurementData: data
				
				MeasurementData {
					id: data
					folder: tab.folder
				}
				
				FourierAnalyzer {
					id:analyzer
					measurementData: data
				}
				
				BandPassAnalyzer {
					id: hpAnalyzer
					measurementData: data
				}
			}
			
			
		}
	}
}
