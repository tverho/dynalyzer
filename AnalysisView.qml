import QtQuick 2.1
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

ColumnLayout {
	id: main
	anchors.fill: parent
	property var analyzer
	
	RowLayout {
		Layout.fillWidth: true
		Label {text: "Folder:"}
		Label {
			text: main.analyzer.folder
			Layout.fillWidth: true
			Layout.preferredWidth: 0
			elide: Text.ElideLeft
		}
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
			Image {
				id: cameraImage
				width: 800
				height: 800
				source: main.analyzer.ready? "image://snapshot/" + navigator.curFrame : ""
				property real selectionX: rect.x
				property real selectionY: rect.y
				property real selectionWidth: rect.width
				property real selectionHeight: rect.height
				
				Rectangle {
					id: rect
					color: "transparent"
					border.width: 2
					border.color: "blue"
				}
				
				MouseArea {
					anchors.fill: parent
					acceptedButtons: Qt.RightButton
					
					onPressed: {
						rect.x = mouse.x;
						rect.y = mouse.y - mouse.y%8;
						rect.width = 0;
						rect.height = 0;
					}
					
					onPositionChanged: {
						rect.width = mouse.x - rect.x
						rect.height = mouse.y-mouse.y%8 - rect.y
					}
				}
				
				Rectangle {
					id: analyzedRegion
					color: "#88000000"
					
					Item {
						id: cross
						property real positionX: (x + analyzedRegion.x) / cameraImage.width
						property real positionY: (y + analyzedRegion.y) / cameraImage.height
						
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
					
					Image {
						id: overlayImage
						anchors.fill: parent
						source: overlayCheckbox.checked && navigator.curFrameAnalyzed ? "image://overlay/"+navigator.curFrame+","+overlayFrequency.text+","+overlayTreshold.text : ""
					}
					
					function set() {
						x = rect.x;
						y = rect.y;
						width = rect.width;
						height = rect.height;
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
					id: overlayFrequency
					text: "1"
				}
			}
			RowLayout {
				Label{
					text: "treshold"
				}
				TextField {
					id: overlayTreshold
					text: "10"
				}
			}
			
			
			Image {
				id:fourierImage
				visible: false
				source: visible? "image://analysis/" + spectrumPos : ""
				property string spectrumPos: cross.x+","+cross.y
			}
		}
	}
}