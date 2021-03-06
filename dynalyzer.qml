import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2

import org.dynalyzer 1.0


ApplicationWindow {
	id: app
	visible: true
	width: 1000
	height: 800
	title: "Dynalyzer"
	
	property url dataPath
	
	onDataPathChanged: {
		/*if (tabview.count) {
			tabview.removeTab(0);
		}*/
		var component = Qt.createComponent("AnalysisView.qml");
		var tab = tabview.addTab(dataPath.toString().split('/').pop());
		component.createObject(tab, {"folder": dataPath});
		tab.active = true;
	}
	
	Component.onCompleted: {
		if (typeof loadpath === 'object') dataPath = loadpath;
	}
	
	ColumnLayout {
		anchors.fill: parent
		
		Button {
			text: "Open measurement"
			
			onClicked: dataPathDialog.open()
			
			FileDialog {
				id: dataPathDialog
				title: "Choose a folder"
				selectFolder: true
				onAccepted: {
					dataPath = folder;
					folder = folder; // Won't remember it otherwise!
				}
			}
		}
	
		TabView {
			id: tabview
			Layout.fillHeight: true
			Layout.fillWidth: true
			
		}
	}
}
