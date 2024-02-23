#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

using namespace CLHEP;

int main(){
  // Define the dimensions of the cube
  G4double halfLengthX = 1.0*cm;
  G4double halfLengthY = 1.0*cm;
  G4double halfLengthZ = 1.0*cm;

  // Create the solid volume for the cube
  G4Box* solidVolume = new G4Box("Cube", halfLengthX, halfLengthY, halfLengthZ);

  // Create the logical volume for the cube
  G4LogicalVolume* logicalVolume = new G4LogicalVolume(solidVolume, 0, "Cube");

  // Create the physical volume for the cube
  G4VPhysicalVolume* physicalVolume = new G4PVPlacement(0, G4ThreeVector(), logicalVolume, "Cube", 0, false, 0);

  // Set the visualization attributes for the cube
  G4VisAttributes* visAttributes = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0)); // Red color
  visAttributes->SetVisibility(true);
  // logicalVolume->SetVisAttributes(visAttributes);
}
