import QtQuick 2.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

import org.dynalyzer 1.0

ColumnLayout {
	id: main
	property FourierAnalyzer analyzer
	
	RowLayout {
		Button {
			text: "Browse"
			onClicked: dataPathDialog.open()
			
			FileDialog {
				id: dataPathDialog
				title: "Choose a folder"
				selectFolder: true
				onAccepted: main.analyzer.folder = folder;
			}
		}
		
		Button {
			text: "Analyze"
			
			onClicked: {
				var x = cameraImage.selectionX;
				var y = cameraImage.selectionY;
				var width = cameraImage.selectionWidth;
				var height = cameraImage.selectionHeight;
				var tstart = navigator.selectionStart;
				var twindow = navigator.selectionEnd - tstart;
				main.analyzer.analyze(x, y, tstart, width, height, twindow);
				analyzedRegion.set();
				analyzedInterval.set();
				fourierImage.visible = true;
			}
		}
	}
	
	RowLayout {
		ColumnLayout {
			SnapshotView {
				id: cameraImage
				width: 400
				height: 400
				analyzer: main.analyzer
				frame: navigator.curFrame
				property int selectionX
				property int selectionY
				property int selectionWidth
				property int selectionHeight
				
				
				Rectangle {
					id: selectionRect
					x: cameraImage.selectionX
					y: cameraImage.selectionY
					width: cameraImage.selectionWidth
					height: cameraImage.selectionHeight
					color: "transparent"
					border.width: 2
					border.color: "blue"
				}
				
				MouseArea {
					anchors.fill: parent
					acceptedButtons: Qt.RightButton
					
					onPressed: {
						cameraImage.selectionX = mouse.x;
						cameraImage.selectionY = mouse.y - mouse.y%8;
						cameraImage.selectionWidth = 0;
						cameraImage.selectionHeight = 0;
					}
					
					onPositionChanged: {
						cameraImage.selectionWidth = mouse.x - cameraImage.selectionX;
						cameraImage.selectionHeight = mouse.y-mouse.y%8 - cameraImage.selectionY;
					}
				}
				
				Rectangle {
					id: analyzedRegion
					color: "#66000000"
					
					OverlayImage {
						id: overlayImage
						visible: overlayCheckbox.checked
						anchors.fill: parent
						analyzer: main.analyzer
						frame: navigator.curFrame
						frequency: overlayFrequencyField.text
						treshold: overlayTresholdField.text
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
						x = cameraImage.selectionX;
						y = cameraImage.selectionY;
						width = cameraImage.selectionWidth;
						height = cameraImage.selectionHeight;
					}
				}
			}
			
			
		}
		
		ColumnLayout {
			CheckBox {
				id: overlayCheckbox
				text: "Frequency overlay"
				checked: false
			}
			RowLayout {
				Label {
					text: "frequency"
				}
				
				TextField {
					id: overlayFrequencyField
					validator: IntValidator {bottom: 0}
					text: "1"
				}
			}
			RowLayout {
				Label{
					text: "treshold"
				}
				TextField {
					id: overlayTresholdField
					validator: IntValidator {bottom: 0}
					text: "10"
				}
			}
			
			SpectrumImage {
				id:fourierImage
				visible: false
				width: 800
				height: 600
				analyzer: main.analyzer
				targetX: cross.x
				targetY: cross.y
			}
		}
	}
	
	ColumnLayout {
		id: navigator
		property int curFrame: 0
		property int numFrames: main.analyzer.ready? main.analyzer.nFrames : 0
		
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
			Layout.minimumWidth: 1000
			color: "white"
			border.color: "black"
			
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
				anchors.fill: parent
				acceptedButtons: Qt.LeftButton
				
				onPressed: navigator.curFrame = navigator.numFrames * mouse.x / slider.width
				onPositionChanged: {
					if (containsMouse)
						navigator.curFrame = navigator.numFrames * mouse.x / slider.width
				}
			}
			
			MouseArea {
				anchors.fill: parent
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
				
				MouseArea {
					anchors.fill: parent
					onWheel: navigator.setFrame(navigator.curFrame + (wheel.angleDelta.y > 0? 5 : -5))
				}
			}
			
			Button {
				text: "next"
				onClicked: navigator.setFrame(navigator.curFrame+1)
			}
		}
	}
}