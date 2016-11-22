import QtQuick 2.1
import QtQuick.Controls 1.1

TextField { 
	property string value: text
	
	onTextChanged: if (acceptableInput) value = text;
	onFocusChanged: if (!focus) text = value
}
