import QtQuick 2.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

// context property: analyzer


ApplicationWindow {
	id: app
	visible: true
	title: "Dynalyzer"
		
	TabView {
		id: tabview
		anchors.fill: parent
	}	
	Component {
		id: tab
		property var analyzer
		Tab {
			title: view.analyzer.folder
			AnalysisView {
				analyzer: tab.analyzer
			}
		}
	}
		
	
	Component.onCompleted:{
		var analyzer = FourierAnalyzer("/home/tuukka/measurement data/dynamat/20160305 Molmat/20160503_0425");
		tab.analyzer = analyzer;
		tabview.addTab(tab);
}
