import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts

ApplicationWindow {
    visible: true
    title: "Virtual camera control center (VCCC)"
    property int margin: 11
    width: (mainLayout.implicitWidth + (2 * margin))
    height: (mainLayout.implicitHeight + (2 * margin))
    minimumWidth: mainLayout.Layout.implicitWidth + margin * 2
    minimumHeight: mainLayout.Layout.implicitHeight + margin * 2

    property QtObject enumerator
    property QtObject reader

    Connections {
        target: cameraSelector
        function onCurrentIndexChanged() {
            console.log("Selected camera: " + cameraSelector.model[cameraSelector.currentIndex])
            // enumerator.setUsedCamera(cameraSelector.model[cameraSelector.currentIndex])
            reader.set_camera(cameraSelector.model[cameraSelector.currentIndex])
        }
    }
    Connections {
        target: enumerator
        function onUpdated(cameras) {
            
            cameraSelector.model = ["None"].concat(cameras);
        }
    }

    ColumnLayout {
        id: mainLayout
        anchors.fill: parent
        anchors.margins: margin
        GroupBox {
            id: rowBox
            title: "Camera"
            Layout.fillWidth: true

            RowLayout {
                id: rowLayout
                anchors.fill: parent
                Text { text: "Select camera:"; }
                ComboBox {
                    id: cameraSelector
                    Layout.fillWidth: true
                    model: ["None"]
                }
            }
        }
        GroupBox {
            title: "Effects"
            Layout.fillWidth: true

            GridLayout {
                columns: 4
                columnSpacing: 15
                rowSpacing: 10
                Repeater {
                    model: ["None", "Sepia", "RedEye", "Star Wars", "Snowfall", "Mirror in the middle", "Hat", "Stylization"]
                    RadioButton {
                        text: modelData
                        checked: modelData === "None"
                        onClicked: {
                            if (checked) {
                                console.log("Clicked " + modelData)
                                reader.set_effect(modelData)
                            }
                        }
                    }
                }
            }
        }
        Item {
            // spacer
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}