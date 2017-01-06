import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2
 
Dialog {
	id: dialog
	title: "Export image series"
	//visible: true
	
	property bool range: rangeButton.checked
	property var frames: range? [startField.text, endField.text, stepField.text] : listField.text.split(',')
	property url folder: folderDialog.folder
	property bool addAnalogSignalPlot: analogSignalPlotCheckbox.checked
	
	standardButtons: StandardButton.Save | StandardButton.Cancel
	
	function setRange(start, end) {
		startField.text = start;
		endField.text = end;
	}
	
	ColumnLayout {
		ExclusiveGroup {id: modeGroup }
		RadioButton {
			id: rangeButton
			text: "Range"
			exclusiveGroup: modeGroup
			checked: true
		}
		ColumnLayout {
			enabled: rangeButton.checked
			RowLayout {
				Label {text: "Start" ; Layout.minimumWidth: 50}
				TextField {
					id: startField
					text: "0"
					validator: IntValidator {bottom: 0}
				}
			}
			RowLayout {
				Label {text: "End" ; Layout.minimumWidth: 50}
				TextField {
					id: endField
					text: "0"
					validator: IntValidator {bottom: 0}
				}
			}
			RowLayout {
				Label {text: "Step" ; Layout.minimumWidth: 50}
				TextField {
					id: stepField
					text: "10"
					validator: IntValidator {bottom: 0}
				}
			}
			CheckBox {
				id: analogSignalPlotCheckbox
				text: "Add force-time plot"
				checked: false
			}
		}
		RadioButton {
			id: listButton
			text: "List of frames (comma separated)"
			exclusiveGroup: modeGroup
		}
		TextArea {
			enabled: listButton.checked
			id: listField
			Layout.minimumWidth: 200
			implicitHeight: 50
			
		}
		
		RowLayout {
			Label {
				text: ("" + dialog.folder).replace('file://', '')
				elide: Text.ElideLeft
				Layout.maximumWidth: 200
			}
			Button {
				text: "..."
				implicitWidth: 30
				onClicked: folderDialog.open()
			}
				
			FileDialog {
				id: folderDialog
				title: "Choose a folder"
				selectFolder: true
				folder: '.'
				onAccepted: {
					folder = folder; // Won't remember it otherwise!
				}
			}
		}
			
	}
}
