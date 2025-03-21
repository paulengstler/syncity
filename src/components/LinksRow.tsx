import { Button, Wrap, WrapItem, Box, useColorModeValue, Show, useDisclosure, Collapse, Code } from '@chakra-ui/react'
import { ExternalLinkIcon, ChevronDownIcon, ChevronUpIcon, Icon } from '@chakra-ui/icons'
import { AiOutlineGithub } from "react-icons/ai"
import NextLink from 'next/link'

import { title, abstract, citationId, citationAuthors, citationYear, citationBooktitle, acknowledgements } from 'data'

import { mode } from "@chakra-ui/theme-tools"
import { links } from 'data'

import { createIcon } from "@chakra-ui/react";

function renderCitationToggle() {
  const { isOpen, onToggle } = useDisclosure()

  function renderButton(isOpen: boolean) {
    if (isOpen) {
      return (
        <>
          <Button size='md' rightIcon={<ChevronUpIcon />}  isActive={isOpen} onClick={onToggle} fontFamily={"IBM Plex Sans"}>Citation</Button>
        </>
      )
    } else {
      return (
        <>
          <Button size='md' rightIcon={<ChevronDownIcon />} onClick={onToggle} fontFamily={"IBM Plex Sans"}>Citation</Button>
        </>
      )
    }
  }
  
  return (
    <>
    <Box>
    <Wrap pt="2rem" pb="2rem">
      <WrapItem>
      <NextLink href={links.paper} passHref={true} target="_blank">
        <Button size='md' fontFamily={"IBM Plex Sans"}>
          arXiv <ExternalLinkIcon ml="0.5rem" />
        </Button>
      </NextLink>
      </WrapItem>
      <WrapItem>
      <NextLink href={links.github} passHref={true} target="_blank">
        <Button leftIcon={<Icon as={AiOutlineGithub}/>} size='md' fontFamily={"IBM Plex Sans"}>
          Code
        </Button>
      </NextLink>
      </WrapItem>
      <WrapItem>
      <NextLink href="#interactive-demo">
        <Button size='md' fontFamily={"IBM Plex Sans"}>
          Interactive Demo
        </Button>
      </NextLink>
      </WrapItem>
      <WrapItem>
    {renderButton(isOpen)}
    </WrapItem>
    </Wrap>
    <Collapse in={isOpen} animateOpacity>
      <Code p="0.5rem" mb="1rem" borderRadius="5px" overflow="scroll" whiteSpace="normal" backgroundColor={useColorModeValue('"#cedeee"', '#3a4557')}>  {/*  fontFamily="monospace" */}
          @inproceedings&#123; <br />
            &nbsp;&nbsp;&nbsp;&nbsp;{citationId}, <br />
            &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;{title}&#125; <br />
            &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;{citationAuthors}&#125; <br />
            &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;{citationYear}&#125; <br />
            &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;{citationBooktitle}&#125; <br />
          &#125;
        </Code>
    </Collapse>
    </Box>
    </>
  )
}

export const LinksRow = () => (
  renderCitationToggle()
)

