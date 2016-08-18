import QtQuick 2.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

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
				
				property int selectionX
				property int selectionY
				property int selectionWidth
				property int selectionHeight
				
				SnapshotView {
					anchors.fill: parent
					measurementData: measurementData
					frame: parent.frame
					scale: parent.scaleFactor
					transformOrigin: Item.TopLeft
				}
				
				Rectangle {
					id: selectionRect
					x: cameraView.selectionX *cameraView.scaleFactor
					y: cameraView.selectionY *cameraView.scaleFactor
					width: cameraView.selectionWidth *cameraView.scaleFactor
					height: cameraView.selectionHeight *cameraView.scaleFactor
					color: "transparent"
					border.width: 2
					border.color: "blue"
				}
				
				MouseArea {
					anchors.fill: parent
					acceptedButtons: Qt.RightButton
					
					onPressed: {
						cameraView.selectionX = mouse.x / cameraView.scaleFactor;
						cameraView.selectionY = mouse.y / cameraView.scaleFactor;
						cameraView.selectionY -= cameraView.selectionY % 8;
						cameraView.selectionWidth = 0;
						cameraView.selectionHeight = 0;
					}
					
					onPositionChanged: if (containsMouse) {
						cameraView.selectionWidth = mouse.x/cameraView.scaleFactor - cameraView.selectionX;
						cameraView.selectionHeight = mouse.y/cameraView.scaleFactor - cameraView.selectionY;
						cameraView.selectionHeight -= cameraView.selectionHeight % 8;
					}
				}
				
				Rectangle {
					id: analyzedRegion
					visible: false
					color: "#66000000"
					property int dataX
					property int dataY
					property int dataWidth
					property int dataHeight
					property int cursorX: cross.x / scale
					property int cursorY: cross.y / scale
					x: dataX * cameraView.scaleFactor
					y: dataY * cameraView.scaleFactor
					width: dataWidth * cameraView.scaleFactor
					height: dataHeight * cameraView.scaleFactor
					
					BPFOverlayImage {
						id: overlayImage
						visible: overlayCheckbox.checked
						anchors.fill: parent
						analyzer: bpAnalyzer
						frame: navigator.curFrame
						//frequency: overlayFrequencyField.text
						treshold: overlayTresholdField.text != ""? overlayTresholdField.text : 999999
						scale: cameraView.scaleFactor
						transformOrigin: Item.TopLeft
					}
					
					Item {
						id: cross
						
						Rectangle {
							x: -3
							width: 8
							height: 2
						}
						Rectangle {
							y: -3
							height: 8
							width: 2
						}
					}
					
					MouseArea {
						width: parent.width - 1
						height: parent.height - 1
						acceptedButtons: Qt.LeftButton
						
						onPressed: {
							cross.x = mouse.x;
							cross.y = mouse.y;
						}
						onPositionChanged: {
							if (containsMouse) {
								cross.x = mouse.x;
								cross.y = mouse.y;
							}
						}
						
					}
					
					function set() {
						visible = true;
						dataX = cameraView.selectionX;
						dataY = cameraView.selectionY;
						dataWidth = cameraView.selectionWidth;
						dataHeight = cameraView.selectionHeight;
					}
				}
			}
			
			
		}
		
		ColumnLayout {
			RowLayout {
				Label {
					text: "BPF low (Hz)"
				}
				
				TextField {
					id: bpfLowFrequencyField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "50"
				}
				
				Label {
					text: "BPF high (Hz)"
				}
				
				TextField {
					id: bpfHighFrequencyField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "100"
				}
			}
			
			CheckBox {
				id: overlayCheckbox
				text: "Overlay"
				checked: true
			}
			/*RowLayout {
				Label {
					text: "frequency"
				}
				
				TextField {
					id: overlayFrequencyField
					validator: IntValidator {bottom: 0}
					text: "1"
				}
			}*/
			RowLayout {
				Label{
					text: "Treshold"
				}
				TextField {
					id: overlayTresholdField
					validator: DoubleValidator {bottom: 0; locale: "en"}
					text: "10"
				}
			}
			
			CheckBox {
				id: fftCheckbox
				text: "FFT"
				checked: false
			}
			CheckBox {
				id: hpfCheckbox
				text: "HPF"
				checked: true
			}
			
			Button {
				text: "Analyze"
				enabled: cameraView.selectionWidth > 0 && cameraView.selectionHeight > 0 && 
					navigator.selectionEnd > navigator.selectionStart
				
				onClicked: {
					var x = cameraView.selectionX;
					var y = cameraView.selectionY;
					var width = cameraView.selectionWidth;
					var height = cameraView.selectionHeight;
					var tstart = navigator.selectionStart;
					var twindow = navigator.selectionEnd - tstart;
					bpAnalyzer.lowerLimit = bpfLowFrequencyField.text;
					bpAnalyzer.upperLimit = bpfHighFrequencyField.text;
					if (hpfCheckbox.checked)
						bpAnalyzer.analyze(x, y, tstart, width, height, twindow);
					if (fftCheckbox.checked)
						fourierAnalyzer.analyze(x, y, tstart, width, height, twindow);
						fourierImage.visible = true;
					analyzedRegion.set();
					analyzedInterval.set();
					
				}
			}
			
			Button {
				text: "Save video"
				
				onClicked: {
					exporter.saveVideoFrames(main.hpAnalyzer, "/home/tuukka/tmp/dynamat_video", 10)
				}
				
				VideoExporter {
					id:exporter
				}
			}
		}
		
		ColumnLayout {
			SpectrumImage {
				id:fourierImage
				visible: false
				width: 400
				height: 300
				analyzer: fourierAnalyzer
				valueCutoff: cutoffSlider.value
				targetX: analyzedRegion.cursorX / cameraView.scaleFactor
				targetY: analyzedRegion.cursorY / cameraView.scaleFactor
			}
			
			Slider {
				id: cutoffSlider
				visible: fourierImage.visible
				value: .25
				minimumValue: 0.0001
			}
		}
	}
	
	ColumnLayout {
		id: navigator
		Layout.fillWidth: true
		
		property int curFrame: 0
		property int numFrames: measurementData && measurementData.ready? measurementData.nFrames : 0
		
		property int selectionStart: 0
		property int selectionEnd: 0
		
		property bool curFrameAnalyzed: curFrame >= analyzedInterval.analysisStart && 
				curFrame <= analyzedInterval.analysisEnd
		
		function setFrame(frame) {
			if (frame < 0) curFrame = 0;
			else if (frame > numFrames-1) curFrame = numFrames - 1;
			else curFrame = frame;
		}
		
		Rectangle {
			id: slider
			Layout.minimumHeight: 50
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
			
			Rectangle {
				id: analyzedInterval
				height: parent.height
				color: "#88000000"
				property int analysisStart
				property int analysisEnd
				x: analysisStart * slider.width / navigator.numFrames
				width: (analysisEnd - analysisStart) * slider.width / navigator.numFrames
				
				function set() {
					analysisStart = navigator.selectionStart;
					analysisEnd = navigator.selectionEnd;
				}
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
		}
	}
	
	FourierAnalyzer {
		id: fourierAnalyzer
		measurementData: measurementData
	}
	
	BandPassAnalyzer {
		id: bpAnalyzer
		measurementData: measurementData
	}
	
	MeasurementData {
		id: measurementData
		folder: main.folder
	}
}