import QtQuick 2.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

import org.dynalyzer 1.0
// context property: analyzer


ApplicationWindow {
	id: app
	visible: true
	title: "Dynalyzer"
		
	TabView {
		id: tabview
		width: 1200
		height: 600
	
		Tab {
			id: tab
			title: folder
			property string folder: "/home/tuukka/measurement data/dynamat/20160305 Molmat/20160503_0425"
			
			AnalysisView {
				analyzer: analyzer
				
				FourierAnalyzer {
					id:analyzer
					folder: tab.folder
				}
			}
			
			
		}
	}
}
