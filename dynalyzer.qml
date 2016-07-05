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
					var component = Qt.createComponent("AnalysisView.qml");
					var tab = tabview.addTab(folder);
					component.createObject(tab, {"folder": folder});
 					
				}
			}
		}
	
		TabView {
			id: tabview
			Layout.fillHeight: true
			
		}
	}
}
