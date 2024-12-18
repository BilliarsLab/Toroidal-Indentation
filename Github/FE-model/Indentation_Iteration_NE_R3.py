# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2018 replay file
# Internal Version: 2017_11_07-12.21.41 127140
# Run by hashourichoshali on Fri Apr  2 19:38:24 2021
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
#: Warning: Permission was denied for "abaqus.rpy"; "abaqus.rpy.849" will be used for this session's replay file.
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=136.888885498047, 
    height=219.733337402344)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
import os
import os.path
#os.chdir(r"C:\Temp")
project_path=r"C:\Users\jli16\Downloads\From_habibeh\Identation_test\R3_Indentation_AI"
#project_path=r"/home/jkingsley/examples/abaqus/Indentation_AI"
os.chdir(project_path)

num_cpus = 16

# parameters
rs = 2   # smaller radius of paper clip
rbrs = 3.25   #Large radius divided by the smaller radius of the indenter
rb = rbrs*rs  #larger radius
LI = 2*rb   #Length of the indenter
krd = 1    # Indentation depth/smaller radius of paper clip
delta = -krd*rs   #Indentation

xx = 20*rs
yy = 15*rs
zz = 5*rs

seedsize =  1    #Mesh factor
seedsizeI=0.57

maxmesh = 3 * rs/2
minmesh = rs/2


#Jobname = "Myjob4"

s = mdb.models['Model-1'].ConstrainedSketch(name='__sweep__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=(rb, LI), point2=(rb, 0.0))
s.VerticalConstraint(entity=g[2], addUndoState=False)
s.ArcByCenterEnds(center=(0.0, 0.0), point1=(rb, 0.0), point2=(-rb, 0.0), 
    direction=CLOCKWISE)
s.Line(point1=(-rb, 0.0), point2=(-rb, LI))
s.VerticalConstraint(entity=g[4], addUndoState=False)
s.unsetPrimaryObject()
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0, transform=(-1.0, 0.0, 0.0, 0.0, -0.0, 1.0, 0.0, 1.0, 0.0, 
    rb, LI, 0.0))
g1, v1, d1, c1 = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=SUPERIMPOSE)
s1.ConstructionLine(point1=(-100.0, 0.0), point2=(100.0, 0.0))
s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
s1.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(rs, 0.0))
p = mdb.models['Model-1'].Part(name='Indenter', dimensionality=THREE_D, 
    type=DISCRETE_RIGID_SURFACE)
p = mdb.models['Model-1'].parts['Indenter']
p.BaseShellSweep(sketch=s1, path=s)
s1.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Indenter']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
del mdb.models['Model-1'].sketches['__sweep__']

p = mdb.models['Model-1'].parts['Indenter']
p.ReferencePoint(point=(0.0, -(rs+rb), 0.0))
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v1, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(-xx, -yy), point2=(xx, yy))
p = mdb.models['Model-1'].Part(name='Substrate', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Substrate']
p.BaseSolidExtrude(sketch=s, depth=zz)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Substrate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']



p = mdb.models['Model-1'].parts['Substrate']
p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
p = mdb.models['Model-1'].parts['Substrate']
p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0.0)
p = mdb.models['Model-1'].parts['Substrate']
c = p.cells
pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
d = p.datums
p.PartitionCellByDatumPlane(datumPlane=d[2], cells=pickedCells)
p = mdb.models['Model-1'].parts['Substrate']
c = p.cells
pickedCells = c.getSequenceFromMask(mask=('[#3 ]', ), )
d1 = p.datums
p.PartitionCellByDatumPlane(datumPlane=d1[3], cells=pickedCells)



a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['Indenter']
a.Instance(name='Indenter-1', part=p, dependent=ON)
p = mdb.models['Model-1'].parts['Substrate']
a.Instance(name='Substrate-1', part=p, dependent=ON)

a = mdb.models['Model-1'].rootAssembly
a.rotate(instanceList=('Indenter-1', ), axisPoint=(0.0, 0.0, 0.0), 
    axisDirection=(1.0, 0.0, 0.0), angle=90.0)

a = mdb.models['Model-1'].rootAssembly
r1 = a.instances['Indenter-1'].referencePoints
v1 = a.instances['Substrate-1'].vertices
a.CoincidentPoint(movablePoint=r1[2], fixedPoint=v1[1])

a = mdb.models['Model-1'].rootAssembly
p = a.instances['Indenter-1']
p.ConvertConstraints()
#: All position constraints of "Indenter-1" have been converted to absolute positions

#
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial', 
    maxNumInc=200, initialInc=0.0015, minInc=1e-09, maxInc=0.05, nlgeom=ON)
    
    
#Mesh
p = mdb.models['Model-1'].parts['Substrate']
p.seedPart(size=seedsize, deviationFactor=0.1, minSizeFactor=0.1)
  
p = mdb.models['Model-1'].parts['Substrate']
e = p.edges
pickedEdges1 = e.getSequenceFromMask(mask=('[#1000004 ]', ), )
pickedEdges2 = e.getSequenceFromMask(mask=('[#90820841 ]', ), )
p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
    end2Edges=pickedEdges2, minSize=minmesh, maxSize=maxmesh, constraint=FINER)


p = mdb.models['Model-1'].parts['Substrate']
e = p.edges
pickedEdges1 = e.getSequenceFromMask(mask=('[#81400 ]', ), )
pickedEdges2 = e.getSequenceFromMask(mask=('[#250000 ]', ), )
p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
    end2Edges=pickedEdges2, minSize=minmesh, maxSize=maxmesh, constraint=FINER)
p = mdb.models['Model-1'].parts['Substrate']
e = p.edges
pickedEdges1 = e.getSequenceFromMask(mask=('[#2008080 ]', ), )
pickedEdges2 = e.getSequenceFromMask(mask=('[#20002200 ]', ), )
p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
    end2Edges=pickedEdges2, minSize=minmesh, maxSize=maxmesh, constraint=FINER)
p = mdb.models['Model-1'].parts['Substrate']
e = p.edges
pickedEdges1 = e.getSequenceFromMask(mask=('[#8000120 ]', ), )
pickedEdges2 = e.getSequenceFromMask(mask=('[#100010 #1 ]', ), )
p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
    end2Edges=pickedEdges2, minSize=minmesh, maxSize=maxmesh, constraint=FINER)
p = mdb.models['Model-1'].parts['Substrate']
e = p.edges
pickedEdges1 = e.getSequenceFromMask(mask=('[#40400002 ]', ), )
pickedEdges2 = e.getSequenceFromMask(mask=('[#4004008 ]', ), )
p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
    end2Edges=pickedEdges2, minSize=minmesh, maxSize=maxmesh, constraint=FINER)


p = mdb.models['Model-1'].parts['Substrate']
p.generateMesh()


p = mdb.models['Model-1'].parts['Indenter']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p = mdb.models['Model-1'].parts['Indenter']
p.seedPart(size=seedsizeI, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Model-1'].parts['Indenter']
p.generateMesh()

#Interaction
mdb.models['Model-1'].ContactProperty('Friction')
mdb.models['Model-1'].interactionProperties['Friction'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    0.5, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)
#: The interaction property "Friction" has been created.
session.viewports['Viewport: 1'].view.setValues(nearPlane=278.252, 
    farPlane=511.978, width=149.245, height=106.84, viewOffsetX=9.02197, 
    viewOffsetY=-2.72405)
a = mdb.models['Model-1'].rootAssembly
s1 = a.instances['Indenter-1'].faces
side2Faces1 = s1.getSequenceFromMask(mask=('[#2 ]', ), )
region1=a.Surface(side2Faces=side2Faces1, name='m_Surf-1')
a = mdb.models['Model-1'].rootAssembly
s1 = a.instances['Substrate-1'].faces
side1Faces1 = s1.getSequenceFromMask(mask=('[#41050 ]', ), )
region2=a.Surface(side1Faces=side1Faces1, name='s_Surf-1')

mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='Contact', 
    initialClearance=OMIT, datumAxis=None, clearanceRegion=None,
    createStepName='Initial', master=region1, slave=region2, sliding=FINITE, 
    enforcement=NODE_TO_SURFACE, thickness=OFF, interactionProperty='Friction', 
    surfaceSmoothing=NONE, adjustMethod=NONE, smooth=0.2)


a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Substrate-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#80484 ]', ), )
region = a.Set(faces=faces1, name='Set-1')
mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
    region=region, u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET, 
    amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)

a = mdb.models['Model-1'].rootAssembly
r1 = a.instances['Indenter-1'].referencePoints
refPoints1=(r1[2], )
region = a.Set(referencePoints=refPoints1, name='Set-2')
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial', 
    region=region, u1=SET, u2=SET, u3=UNSET, ur1=SET, ur2=SET, ur3=SET, 
    amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)

a = mdb.models['Model-1'].rootAssembly
r1 = a.instances['Indenter-1'].referencePoints
refPoints1=(r1[2], )
region = a.Set(referencePoints=refPoints1, name='Set-3')
mdb.models['Model-1'].DisplacementBC(name='BC-3', createStepName='Step-1', 
    region=region, u1=UNSET, u2=UNSET, u3=delta, ur1=UNSET, ur2=UNSET, 
    ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, 
    fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)


import numpy as np

list_values = np.concatenate([
    np.linspace(0.1, 1, 10),
    np.linspace(1, 10, 15),
    np.linspace(10, 100, 30),
    np.linspace(100, 200, 20)
])

# Remove duplicated values
list_values = np.unique(list_values) 

counter = 0
#v1 = 0.4
#v2 = 0.4
for E1 in list_values:  # kpa 1,5,10,30,50,70,90,100
    for E2 in list_values:  # kpa
        for v12 in [0.45]:
            v23_t = 1 - float(E2) / (2 * E1)  # Ensure floating-point division
            for v23 in [v23_t]:
                G12_t = float(E2) * E1 / (E2 * (1 + 2 * 0.45) + E1)  # Ensure floating-point division
                for G12 in [G12_t]:
                    if (E1 > E2) and (E1 < 100 * E2):
                        G23 = float(E2) / (2 * (1 + v23))  # Ensure floating-point division
                        counter = int(counter + 1)
                        fname = str(counter) + "_X.txt"
                        if not (os.path.isfile(fname)):# and (E1/G12 <50):
                            Jobname = "Job_X3_" + str(counter)
                            #............................Material
                            mdb.models['Model-1'].Material(name='Substrate')
                            mdb.models['Model-1'].materials['Substrate'].Elastic(
                                type=ENGINEERING_CONSTANTS, table=((E1, E2, E2, v12, v12, v23, G12, G12, 
                                G23), ))
                            mdb.models['Model-1'].HomogeneousSolidSection(name='SubSec', 
                                material='Substrate', thickness=None)
                                
                            p = mdb.models['Model-1'].parts['Substrate']
                            c = p.cells
                            cells = c.getSequenceFromMask(mask=('[#f ]', ), )
                            region = p.Set(cells=cells, name='Set-1')
                            p = mdb.models['Model-1'].parts['Substrate']
                            p.SectionAssignment(region=region, sectionName='SubSec', offset=0.0, 
                                offsetType=MIDDLE_SURFACE, offsetField='', 
                                thicknessAssignment=FROM_SECTION)
                            p = mdb.models['Model-1'].parts['Substrate']
                            c = p.cells
                            cells = c.getSequenceFromMask(mask=('[#f ]', ), )
                            region = regionToolset.Region(cells=cells)
                            orientation=None
                            mdb.models['Model-1'].parts['Substrate'].MaterialOrientation(region=region, 
                                orientationType=GLOBAL, axis=AXIS_1, additionalRotationType=ROTATION_NONE, 
                                localCsys=None, fieldName='', stackDirection=STACK_3)


                            #Job creation, submission
                            mdb.Job(name=Jobname, model='Model-1', description='', type=ANALYSIS, 
                                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
                                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
                                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=8, 
                                numDomains=8, numGPUs=0)

                            mdb.jobs[Jobname].submit(consistencyChecking=OFF)
                            mdb.jobs[Jobname].waitForCompletion()

                            #.................................................Post processing

                            from odbAccess import *
                            from odbMaterial import *
                            from odbSection import *

                            out1 = open(str(counter)+'_X.txt', 'w')
                            out1.write(str(E1)+";"+str(E2) + ";"+str(v12)+ ";"+str(v23)+ ";"+str(G12))
                            out1.write('\n')

                            a = mdb.models['Model-1'].rootAssembly
                            session.viewports['Viewport: 1'].setValues(displayedObject=a)
                            session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
                                predefinedFields=OFF, connectors=OFF)
                            #o3 = openOdb(path=Job_Name+'.odb')
                            o3 = session.openOdb(
                                name=project_path+'/'+Jobname+'.odb')
                            session.viewports['Viewport: 1'].setValues(displayedObject=o3)
                            session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
                                CONTOURS_ON_DEF, ))

                            session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
                                variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 
                                'S11'), )		
                            #odb = session.odbs[Jobname+'.odb']
                            odb = session.odbs[project_path+'/'+Jobname+'.odb']

                            OutputU = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
                                NODAL, ((INVARIANT, 'Magnitude'), )), ), nodeSets=("SET-2", ))

                            OutputF = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
                                NODAL, ((INVARIANT, 'Magnitude'), )), ), nodeSets=("SET-2", ))


                            for i in range (0, len(OutputU[0]), 1):
                                out1.write(str(OutputU[0][i][1])+";"+str(OutputF[0][i][1]) )
                                out1.write('\n')

                        #.......................................................90 degree

                            a = mdb.models['Model-1'].rootAssembly
                            a.rotate(instanceList=('Indenter-1', ), axisPoint=(0.0, 0.0, 0.0), 
                                axisDirection=(0.0, 0.0, 1.0), angle=90.0)

                            Jobname = "Job_Y3_" + str(counter)
                            #Job creation, submission
                            mdb.Job(name=Jobname, model='Model-1', description='', type=ANALYSIS, 
                                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
                                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
                                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=num_cpus, 
                                numDomains=num_cpus, numGPUs=0)

                            mdb.jobs[Jobname].submit(consistencyChecking=OFF)
                            mdb.jobs[Jobname].waitForCompletion()

                            a = mdb.models['Model-1'].rootAssembly
                            a.rotate(instanceList=('Indenter-1', ), axisPoint=(0.0, 0.0, 0.0), 
                                axisDirection=(0.0, 0.0, 1.0), angle=90.0)
                            #.................................................Post processing

                            from odbAccess import *
                            from odbMaterial import *
                            from odbSection import *

                            out2 = open(str(counter)+'_Y.txt', 'w')	
                            out2.write(str(E1)+";"+str(E2) + ";"+str(v12)+ ";"+str(v23)+ ";"+str(G12))
                            out2.write('\n')

                            a = mdb.models['Model-1'].rootAssembly
                            session.viewports['Viewport: 1'].setValues(displayedObject=a)
                            session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
                                predefinedFields=OFF, connectors=OFF)
                            #o3 = openOdb(path=Job_Name+'.odb')
                            o3 = session.openOdb(
                                name=project_path+'/'+Jobname+'.odb')
                            session.viewports['Viewport: 1'].setValues(displayedObject=o3)
                            session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
                                CONTOURS_ON_DEF, ))

                            session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
                                variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 
                                'S11'), )		
                            #odb = session.odbs[Jobname+'.odb']
                            odb = session.odbs[project_path+'/'+Jobname+'.odb']

                            OutputU = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
                                NODAL, ((INVARIANT, 'Magnitude'), )), ), nodeSets=("SET-2", ))

                            OutputF = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
                                NODAL, ((INVARIANT, 'Magnitude'), )), ), nodeSets=("SET-2", ))


                            for i in range (0, len(OutputU[0]), 1):
                                out2.write(str(OutputU[0][i][1])+";"+str(OutputF[0][i][1]) )
                                out2.write('\n')


                            out1.close()
                            out2.close()


















