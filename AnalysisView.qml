import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2

import org.dynalyzer 1.0

ColumnLayout {
	id: main
	property string folder
	anchors.fill: parent
		
	RowLayout {
		ColumnLayout {
			Rectangle {
				id: cameraView
				Layout.fillHeight: true
				Layout.minimumWidth: 600
 				color: "transparent"
				property int frame: navigator.curFrame
				property real scaleFactor: Math.min(width/measurementData.image_width, height/measurementData.image_height)
								
				SnapshotView {
					anchors.fill: parent
					measurementData: measurementData
					frame: parent.frame
					scale: parent.scaleFactor
					transformOrigin: Item.TopLeft
				}
				
				DifferenceOverlayImage {
					id: diffOverlayImage
					visible: overlayCheckbox.checked
					anchors.fill: parent
					analyzer: diffAnalyzer
					frame: navigator.curFrame
					treshold: overlayTresholdField.value
					spatialAveraging: spatialAveragingField.value
					scale: cameraView.scaleFactor
					transformOrigin: Item.TopLeft
				}
			}
		}
		
		ColumnLayout {
			id: propertiesPane
			property double labelWidth: 180
			
			property var json: JSON.stringify({
					interval: intervalField.value,
					treshold: overlayTresholdField.value,
					temporalAveraging: temporalAveragingField.value,
					spatialAveraging: spatialAveragingField.value,
					blackTreshold: blackTresholdField.value,
					fastTemporalAveraging: fastTemporalAveragingCheckbox.checked,
					relativeMode: relativeModeCheckbox.checked
				})
			
			CheckBox {
				id: overlayCheckbox
				text: "Overlay"
				checked: true
			}
			
			RowLayout {
				Label {
					text: "Comparison interval"
					Layout.minimumWidth: propertiesPane.labelWidth
				}
				
				ValueField {
					id: intervalField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "20"
				}
			}
			
			RowLayout {
				Label {
					text: "Treshold"
					Layout.minimumWidth: propertiesPane.labelWidth
				}
				ValueField {
					id: overlayTresholdField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "8"
				}
			}
			
			RowLayout {
				
				Label {
					text: "Temporal averaging"
					Layout.minimumWidth: propertiesPane.labelWidth
				}
				
				ValueField {
					id: temporalAveragingField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "10"
				}
			}
			
			
			RowLayout {
				Label {
					text: "Spatial averaging"
					Layout.minimumWidth: propertiesPane.labelWidth
				}
				
				ValueField {
					id: spatialAveragingField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "2"
				}
			}
			
			RowLayout {
				Label {
					text: "Black treshold"
					Layout.minimumWidth: propertiesPane.labelWidth
				}
				
				ValueField {
					id: blackTresholdField
					validator: IntValidator {bottom: 0; locale: "en"}
					text: "20"
				}
			}
			
			CheckBox {
				id: fastTemporalAveragingCheckbox
				text: "Fast temporal averaging"
				checked: true
			}
			
			CheckBox {
				id: relativeModeCheckbox
				text: "Relative mode"
				checked: true
			}
			
			

			Button {
				text: "Save image"

				onClicked: saveImageDialog.open()

				FileDialog {
					id: saveImageDialog
					title: "Save image"
					nameFilters: [ "Image files (*.png *.tiff *.jpg)", "All files (*)" ]
					selectExisting: false
					onAccepted: {
						console.log("Parameters for the saved image:");
						console.log(propertiesPane.json);
						
						exporter.saveImage(measurementData, diffOverlayImage, navigator.curFrame, fileUrl)
						folder = folder; // Won't remember it otherwise!
					}
				}
				
				
			}
			Button {
				text: "Save image series"
				
				onClicked: {
					if (navigator.selectionEnd - navigator.selectionStart > 0)
						saveImageSeriesDialog.setRange(navigator.selectionStart, navigator.selectionEnd);
					saveImageSeriesDialog.open();
				}
					
				
				ExportDialog {
					id: saveImageSeriesDialog
					
					onAccepted: {
						exporter.saveImageSeries(measurementData, diffOverlayImage, frames, folder, range, addAnalogSignalPlot);
						console.log("Parameters for the saved images:");
						console.log(propertiesPane.json);
					}
				}
			}
			Button {
				text: "Save analog signals"
				
				onClicked: saveSignalsDialog.open()

				FileDialog {
					id: saveSignalsDialog
					title: "Save image"
					nameFilters: [ "ASCII files (*.txt)", "All files (*)" ]
					selectExisting: false
					onAccepted: {
						exporter.saveAnalogSignals(measurementData, fileUrl)
						folder = folder; // Won't remember it otherwise!
					}
				}
			}
			
			
			Exporter {
				id: exporter
			}
		}
	}
	
	ColumnLayout {
		id: navigator
		Layout.fillWidth: true
		
		property int curFrame: 0
		property int numFrames: measurementData? measurementData.nFrames : 0
		
		property int selectionStart: 0
		property int selectionEnd: 0
				
		function setFrame(frame) {
			if (frame < 0) curFrame = 0;
			else if (frame > numFrames-1) curFrame = numFrames - 1;
			else curFrame = frame;
		}
		
		Rectangle {
			id: slider
			Layout.minimumHeight: 100
			Layout.fillWidth: true
			color: "white"
			border.color: "black"
			
			MouseArea {
				anchors.fill: parent
				onWheel: navigator.setFrame(navigator.curFrame + (wheel.angleDelta.y > 0? 5 : -5))
			}
			
			AnalogSignalPlot {
				anchors.fill: parent
				measurementData: measurementData
			}
			
			Rectangle {
				id: cursor
				height: parent.height
				width: 2
				x: navigator.curFrame * slider.width / navigator.numFrames
				color: "black"
			}
			
			Rectangle {
				id: selection
				height: parent.height
				color: "blue"
				opacity: 0.5
				x: navigator.selectionStart * slider.width / navigator.numFrames
				width: (navigator.selectionEnd - navigator.selectionStart) * slider.width / navigator.numFrames
			}
			
			MouseArea {
				width: parent.width-1
				height: parent.height
				acceptedButtons: Qt.LeftButton
				
				onPressed: navigator.curFrame = navigator.numFrames * mouse.x / slider.width
				onPositionChanged: {
					if (containsMouse)
						navigator.curFrame = navigator.numFrames * mouse.x / slider.width
				}
			}
			
			MouseArea {
				width: parent.width-1
				height: parent.height
				acceptedButtons: Qt.RightButton
				
				onPressed: navigator.selectionStart = navigator.numFrames * mouse.x / slider.width
				onPositionChanged: {
					if (containsMouse) 
						navigator.selectionEnd = navigator.numFrames * mouse.x / slider.width
				}	
			}
		}
		
		RowLayout {
			Button {
				text: "prev"
				onClicked: navigator.setFrame(navigator.curFrame-1)
			}
			
			TextField {
				text:navigator.curFrame
				validator: IntValidator {bottom: 0; top:navigator.numFrames-1}
				onEditingFinished: navigator.curFrame = text
			}
			
			Button {
				text: "next"
				onClicked: navigator.setFrame(navigator.curFrame+1)
			}
			
			Label {text: "Play speed (f/s)"}
			
			ValueField {
				id: playSpeedField
				text: "10"
				validator: IntValidator {bottom: 0}
				implicitWidth: 60
			}
			
			Button {
				text: playTimer.running? "stop" : "play"
				onClicked: {
					if (playTimer.running) {
						playTimer.stop();
					} else {
						playTimer.start();
					}
					
				}
				
				Timer {
					id: playTimer
					interval: 100
					repeat: true
					property real step: playSpeedField.value/1000 * interval
					
					onTriggered: {
						if (navigator.curFrame + step < navigator.numFrames) {
							navigator.curFrame += step;
						}
						else {
							stop();
						}
					}
				}
			}
			
			Label {
				text: "" + (1000*navigator.curFrame / measurementData.framerate) +
					"ms   (" + measurementData.framerate + " frames/s)"
			}
		}
	}
		
	DifferenceAnalyzer {
		id: diffAnalyzer
		measurementData: measurementData
		
		interval: intervalField.value
		temporalAveraging: temporalAveragingField.value
		blackTreshold: blackTresholdField.value
		relativeMode: relativeModeCheckbox.checked
		fastTemporalAveraging: fastTemporalAveragingCheckbox.checked
	}
	
	MeasurementData {
		id: measurementData
		folder: main.folder
	}
}
